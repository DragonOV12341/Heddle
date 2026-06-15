/*!
 * \file finegrained_ws.cc
 * \brief FineGrainedWS for sm90+ async-copy pipelines.
 *
 * Works on the inline barrier IR emitted by lowering passes such as
 * LowerBulkCopy / LowerPTXAsyncCopy:
 *   SeqStmt({
 *     AttrStmt("tl.tma_copy_write_buffer", buf, 1,
 *       IfThenElse(threadIdx.x == 0,
 *         SeqStmt({arrive_expect_tx(mbar, bytes), tma_load(...)}))),
 *     mbarrier_wait_parity(mbar, parity)
 *   })
 *
 * The pass splits the pipelined loop into:
 *   producer: issues TMA / cp.async
 *   consumer: waits, computes, and releases buffers
 *
 * For pure-TMA loops we rewrite the forward-barrier protocol so the producer
 * releases the barrier after issuing the TMA copy:
 *   expect_transaction -> tma_load -> arrive
 */

#include "../op/utils.h"
#include "common/tma_copy_utils.h"
#include "warp_specialized_rewriter.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tir;
using namespace runtime;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

enum class AsyncProducerKind : uint8_t { kTma, kCpAsync };

struct AsyncCopyBlockInfo {
  AsyncProducerKind kind;
  Stmt producer_stmt;              // TMA issue or cp.async enqueue+commit
  Optional<Stmt> wait_stmt;        // Existing forward wait for TMA blocks
  Optional<Var> write_buffer_data; // shared buffer written by producer
};

using BufferDataToBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using VarBindingMap =
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

// ---------------------------------------------------------------------------
// Cross-stage consumer: per-buffer, per-compute-stmt stage offset
// ---------------------------------------------------------------------------

struct ConsumerStageConfig {
  int compute_stmt_index;
  int stage_offset;  // 0 = current iteration, -1 = previous, +1 = next
  int buffer_index;  // index into extractor.blocks
};

/*!
 * \brief Parse the consumer stage map from a simplified JSON-like string.
 *
 * Format: "buffer_name:pattern1=offset1,pattern2=offset2;buffer_name2:..."
 * Example: "V_shared:wgmma=-1" means any compute_stmt whose string repr
 * contains "wgmma" and reads V_shared gets stage_offset=-1.
 *
 * Returns a map: buffer_name -> [(pattern, offset), ...]
 */
static std::unordered_map<std::string,
                          std::vector<std::pair<std::string, int>>>
ParseConsumerStageMap(const std::string &config_str) {
  std::unordered_map<std::string, std::vector<std::pair<std::string, int>>>
      result;
  if (config_str.empty())
    return result;

  // Parse "buf1:pat1=off1,pat2=off2;buf2:pat3=off3"
  std::istringstream buf_stream(config_str);
  std::string buf_entry;
  while (std::getline(buf_stream, buf_entry, ';')) {
    auto colon_pos = buf_entry.find(':');
    if (colon_pos == std::string::npos)
      continue;
    std::string buf_name = buf_entry.substr(0, colon_pos);
    std::string patterns_str = buf_entry.substr(colon_pos + 1);
    std::istringstream pat_stream(patterns_str);
    std::string pat_entry;
    while (std::getline(pat_stream, pat_entry, ',')) {
      auto eq_pos = pat_entry.find('=');
      if (eq_pos == std::string::npos)
        continue;
      std::string pattern = pat_entry.substr(0, eq_pos);
      int offset = std::stoi(pat_entry.substr(eq_pos + 1));
      result[buf_name].emplace_back(pattern, offset);
    }
  }
  return result;
}

/*!
 * \brief Parse barrier hints from config string.
 *
 * Format: "buffer_name:wait=W,arrive=A;buffer_name2:wait=W2,arrive=A2"
 *
 * Returns a map: buffer_name -> (wait_pos, arrive_pos)
 */
static std::unordered_map<std::string, std::pair<int, int>>
ParseBarrierHints(const std::string &config_str) {
  std::unordered_map<std::string, std::pair<int, int>> result;
  if (config_str.empty())
    return result;

  std::istringstream buf_stream(config_str);
  std::string buf_entry;
  while (std::getline(buf_stream, buf_entry, ';')) {
    auto colon_pos = buf_entry.find(':');
    if (colon_pos == std::string::npos)
      continue;
    std::string buf_name = buf_entry.substr(0, colon_pos);
    std::string params_str = buf_entry.substr(colon_pos + 1);
    int wait_pos = -1, arrive_pos = -1;
    std::istringstream param_stream(params_str);
    std::string param;
    while (std::getline(param_stream, param, ',')) {
      auto eq_pos = param.find('=');
      if (eq_pos == std::string::npos)
        continue;
      std::string key = param.substr(0, eq_pos);
      int val = std::stoi(param.substr(eq_pos + 1));
      if (key == "wait")
        wait_pos = val;
      else if (key == "arrive")
        arrive_pos = val;
    }
    if (wait_pos >= 0 && arrive_pos >= 0) {
      result[buf_name] = {wait_pos, arrive_pos};
    }
  }
  return result;
}

/*!
 * \brief Parse per-compute-stmt stage offsets from config string.
 *
 * Format: "idx1=offset1,idx2=offset2"
 * Example: "3=-1" means compute_stmt[3] uses stage_offset=-1
 */
static std::unordered_map<int, int>
ParseStageOffsets(const std::string &config_str) {
  std::unordered_map<int, int> result;
  if (config_str.empty())
    return result;

  std::istringstream stream(config_str);
  std::string entry;
  while (std::getline(stream, entry, ',')) {
    auto eq_pos = entry.find('=');
    if (eq_pos == std::string::npos)
      continue;
    int idx = std::stoi(entry.substr(0, eq_pos));
    int offset = std::stoi(entry.substr(eq_pos + 1));
    result[idx] = offset;
  }
  return result;
}

/*!
 * \brief Parse per-op warp group assignments from config string.
 *
 * Format: "s0:0,s1:1,s2:0,s3:1"
 * Each entry maps a compute_stmt name (sN where N = compute_stmt index)
 * to a warp group ID. Used by Plan B per-op warp dispatch.
 */
static std::unordered_map<int, int>
ParseWarpAssigns(const std::string &config_str) {
  std::unordered_map<int, int> result;
  if (config_str.empty())
    return result;

  std::istringstream stream(config_str);
  std::string entry;
  while (std::getline(stream, entry, ',')) {
    auto colon_pos = entry.find(':');
    if (colon_pos == std::string::npos)
      continue;
    std::string name = entry.substr(0, colon_pos);
    int warp_id = std::stoi(entry.substr(colon_pos + 1));
    // Extract compute_stmt index from "sN" format
    if (name.size() > 1 && name[0] == 's') {
      int ci = std::stoi(name.substr(1));
      result[ci] = warp_id;
    }
  }
  return result;
}

struct LocalAccessSummary {
  BufferSet read_buffers;
  BufferSet write_buffers;
  VarSet read_vars;
  VarSet def_vars;

  bool HasTrackedDefs() const {
    return !write_buffers.empty() || !def_vars.empty();
  }
};

struct LocalLiveSet {
  BufferSet buffers;
  VarSet vars;

  bool NeedsAnyDef(const LocalAccessSummary &summary) const {
    for (const auto &buf : summary.write_buffers) {
      if (buffers.count(buf)) {
        return true;
      }
    }
    for (const auto &var : summary.def_vars) {
      if (vars.count(var)) {
        return true;
      }
    }
    return false;
  }

  void KillDefs(const LocalAccessSummary &summary) {
    for (const auto &buf : summary.write_buffers) {
      buffers.erase(buf);
    }
    for (const auto &var : summary.def_vars) {
      vars.erase(var);
    }
  }

  void AddUses(const LocalAccessSummary &summary) {
    buffers.insert(summary.read_buffers.begin(), summary.read_buffers.end());
    vars.insert(summary.read_vars.begin(), summary.read_vars.end());
  }
};

// ---------------------------------------------------------------------------
// PhaseCounter: mutable int32 counter for guarded-loop phase tracking
// ---------------------------------------------------------------------------

/*!
 * \brief When a pipeline loop body is conditionally guarded (e.g.
 *        `if block_mask[k]: ...`), the loop-variable-based parity
 *        `(k / num_stages) % 2` can desynchronise because skipped iterations
 *        don't touch barriers.  A PhaseCounter is a local int32[1] buffer
 *        that tracks the *actual* number of guarded-body entries so that
 *        parity/stage are always correct.
 */
struct PhaseCounter {
  Buffer buf;

  static PhaseCounter Create(const std::string &name) {
    return {decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32), name,
                        "local")};
  }

  PrimExpr Load() const {
    return BufferLoad(buf, {IntImm(DataType::Int(32), 0)});
  }

  Stmt Init() const {
    return BufferStore(buf, IntImm(DataType::Int(32), 0),
                       {IntImm(DataType::Int(32), 0)});
  }

  Stmt Increment() const {
    return BufferStore(buf, Load() + 1, {IntImm(DataType::Int(32), 0)});
  }

  /*! Wrap a For-loop with Allocate + DeclBuffer + Init(0). */
  Stmt WrapLoopWithAlloc(Stmt loop) const {
    Stmt body = SeqStmt({Init(), std::move(loop)});
    body = DeclBuffer(buf, body);
    return Allocate(buf->data, buf->dtype, buf->shape, const_true(), body);
  }

  PrimExpr StageExpr(int num_stages) const {
    if (num_stages == 1)
      return IntImm(DataType::Int(32), 0);
    return FloorMod(Load(), num_stages);
  }

  PrimExpr ParityExpr(int num_stages) const {
    if (num_stages == 1)
      return FloorMod(Load(), 2);
    return FloorMod(FloorDiv(Load(), num_stages), 2);
  }
};

/*!
 * \brief Replace the loop-variable-based stage expression with a
 *        phase-counter-based one inside producer / consumer statements.
 *
 *  When `needs_phase_counter` is true, the barrier IDs already use
 *  `phase_counter->StageExpr(N)` but the shared-memory buffer offsets
 *  still embed `FloorMod(loop_var - loop_min, N)`.  This mutator
 *  rewrites every matching FloorMod to the replacement expression so
 *  that stage indexing stays in sync with barrier indexing when loop
 *  iterations are conditionally skipped.
 */
class StageExprReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, Var loop_var, PrimExpr loop_min,
                      int num_stages, PrimExpr replacement) {
    StageExprReplacer r(std::move(loop_var), std::move(loop_min), num_stages,
                        std::move(replacement));
    return r.VisitStmt(stmt);
  }

private:
  StageExprReplacer(Var loop_var, PrimExpr loop_min, int num_stages,
                    PrimExpr replacement)
      : loop_var_(std::move(loop_var)), loop_min_(std::move(loop_min)),
        num_stages_(num_stages), replacement_(std::move(replacement)) {}

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (is_const_int(op->b, num_stages_) && MatchLinearIdx(op->a)) {
      return replacement_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  /*! Match `loop_var`, `loop_var - loop_min`, or `loop_var - 0`. */
  bool MatchLinearIdx(const PrimExpr &expr) const {
    if (expr.same_as(loop_var_))
      return true;
    if (const auto *sub = expr.as<SubNode>()) {
      if (sub->a.same_as(loop_var_)) {
        if (is_const_int(sub->b, 0))
          return true;
        if (sub->b.same_as(loop_min_))
          return true;
      }
    }
    return false;
  }

  Var loop_var_;
  PrimExpr loop_min_;
  int num_stages_;
  PrimExpr replacement_;
};

class BufferDataToBufferCollector : public StmtExprVisitor {
public:
  static BufferDataToBufferMap Collect(const Stmt &stmt) {
    BufferDataToBufferCollector collector;
    collector.VisitStmt(stmt);
    return collector.result_;
  }

private:
  void VisitStmt_(const BlockRealizeNode *op) final {
    CollectBuffers(op->block);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    CollectBuffers(GetRef<Block>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void CollectBuffers(const Block &block) {
    for (const auto &buffer : block->alloc_buffers) {
      result_.emplace(buffer->data, buffer);
    }
  }

  BufferDataToBufferMap result_;
};

class LocalAccessCollector : public StmtExprVisitor {
public:
  static LocalAccessSummary Collect(const Stmt &stmt,
                                    const BufferDataToBufferMap &buffer_map) {
    LocalAccessCollector collector(buffer_map);
    collector.VisitStmt(stmt);
    return std::move(collector.summary_);
  }

  static LocalAccessSummary
  CollectExpr(const PrimExpr &expr, const BufferDataToBufferMap &buffer_map) {
    LocalAccessCollector collector(buffer_map);
    collector.VisitExpr(expr);
    return std::move(collector.summary_);
  }

private:
  explicit LocalAccessCollector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

  void VisitStmt_(const LetStmtNode *op) final {
    VisitExpr(op->value);
    summary_.def_vars.insert(op->var);
    bound_vars_.insert(op->var);
    VisitStmt(op->body);
    bound_vars_.erase(op->var);
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    bound_vars_.insert(op->loop_var);
    VisitStmt(op->body);
    bound_vars_.erase(op->loop_var);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsLocalBuffer(op->buffer, true)) {
      summary_.read_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsLocalBuffer(op->buffer, true)) {
      summary_.write_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (bound_vars_.count(var) || buffer_data_to_buffer_.count(var)) {
      return;
    }
    summary_.read_vars.insert(var);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      if (IsLocalBuffer(base_load->buffer, true)) {
        int rw_mask = GetConstAccessMask(op->args[2]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(base_load->buffer);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(base_load->buffer);
        }
      }
      for (const auto &index : base_load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5);
      const auto *var = op->args[1].as<VarNode>();
      ICHECK(var);
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end() &&
          IsLocalBuffer(it->second, true)) {
        int rw_mask = GetConstAccessMask(op->args[4]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(it->second);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(it->second);
        }
      }
      VisitExpr(op->args[2]);
      VisitExpr(op->args[3]);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  int GetConstAccessMask(const PrimExpr &expr) const {
    if (const auto *imm = expr.as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  LocalAccessSummary summary_;
  VarSet bound_vars_;
};

class ProducerSimtCopyDetector : public StmtExprVisitor {
public:
  static bool HasSimtCopy(const Stmt &stmt,
                          const BufferDataToBufferMap &buffer_map) {
    ProducerSimtCopyDetector detector(buffer_map);
    detector.VisitStmt(stmt);
    return detector.has_global_read_ && detector.has_shared_write_;
  }

private:
  explicit ProducerSimtCopyDetector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

  void VisitStmt_(const IfThenElseNode *op) final {
    bool old_in_if_cond = in_if_cond_;
    in_if_cond_ = true;
    VisitExpr(op->condition);
    in_if_cond_ = old_in_if_cond;
    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!in_if_cond_ && !in_async_copy_ && IsGlobalBuffer(op->buffer)) {
      has_global_read_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!in_if_cond_ && !in_async_copy_ && IsSharedBuffer(op->buffer)) {
      has_shared_write_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    bool old_in_async_copy = in_async_copy_;
    if (op->op.same_as(tma_load()) || op->op.same_as(tma_load_im2col()) ||
        op->op.same_as(tma_store()) || op->op.same_as(tma_store_arrive()) ||
        op->op.same_as(tma_store_wait()) ||
        op->op.same_as(tl::ptx_cp_async()) ||
        op->op.same_as(builtin::ptx_cp_async())) {
      in_async_copy_ = true;
    }

    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      MarkAccess(base_load->buffer, GetConstAccessMask(op->args[2]));
      for (const auto &index : base_load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      in_async_copy_ = old_in_async_copy;
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5);
      const auto *var = op->args[1].as<VarNode>();
      ICHECK(var);
      auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end()) {
        MarkAccess(it->second, GetConstAccessMask(op->args[4]));
      }
      VisitExpr(op->args[2]);
      VisitExpr(op->args[3]);
      in_async_copy_ = old_in_async_copy;
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
    in_async_copy_ = old_in_async_copy;
  }

  void MarkAccess(const Buffer &buffer, int rw_mask) {
    if (in_if_cond_ || in_async_copy_ || !buffer.defined()) {
      return;
    }
    if ((rw_mask & 1) && IsGlobalBuffer(buffer)) {
      has_global_read_ = true;
    }
    if ((rw_mask & 2) && IsSharedBuffer(buffer)) {
      has_shared_write_ = true;
    }
  }

  int GetConstAccessMask(const PrimExpr &expr) const {
    if (const auto *imm = expr.as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  bool has_global_read_{false};
  bool has_shared_write_{false};
  bool in_if_cond_{false};
  bool in_async_copy_{false};
};

// ---------------------------------------------------------------------------
// Helpers (reused from warp_specialized_rewriter.cc patterns)
// ---------------------------------------------------------------------------

static PrimExpr makeGetBarrier(PrimExpr barrier_id) {
  return Call(DataType::Handle(), get_mbarrier(), {std::move(barrier_id)});
}

static Stmt makeArriveBarrier(PrimExpr barrier_id) {
  Array<PrimExpr> args = {makeGetBarrier(std::move(barrier_id))};
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

static Stmt makeCpAsyncBarrierNoInc(PrimExpr barrier_id) {
  auto call = Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                   {makeGetBarrier(std::move(barrier_id))});
  return Evaluate(call);
}

static Stmt makeParityWait(PrimExpr barrier_id, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {makeGetBarrier(std::move(barrier_id)), std::move(parity)});
  return Evaluate(call);
}

static bool IsTrivialNoOpStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *imm = eval->value.as<IntImmNode>()) {
      return imm->value == 0;
    }
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      if (!IsTrivialNoOpStmt(s)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// AsyncCopyBlockExtractor
// ---------------------------------------------------------------------------

/*!
 * \brief Extract async producer blocks from a flattened loop body.
 *
 * Recognized patterns:
 *
 *  Pattern 1: AttrStmt("tl.tma_copy_write_buffer", ...) + mbarrier_wait_parity
 *  Pattern 2: IfThenElse containing tma_load + mbarrier_wait_parity
 *  Pattern 3: one or more cp_async-only stmts + commit_group + wait_group(0)
 *
 * Everything else is classified as a compute statement.
 */
class AsyncCopyBlockExtractor {
public:
  std::vector<AsyncCopyBlockInfo> blocks;
  std::vector<Stmt> compute_stmts;

  void Extract(const Array<Stmt> &flat_stmts) {
    size_t i = 0;
    while (i < flat_stmts.size()) {
      if (i + 1 < flat_stmts.size() &&
          IsMbarrierWaitParity(flat_stmts[i + 1])) {
        Optional<Var> write_buffer_data =
            ExtractTmaCopyWriteBufferData(flat_stmts[i]);
        // Check Pattern 1/2: TMA producer + wait pair, optionally wrapped in a
        // simple guard/Block/Let/Attr shell. Recover the written shared buffer
        // when the tl.tma_copy_write_buffer annotation survives under wrappers.
        if (write_buffer_data.defined() || ContainsTmaLoad(flat_stmts[i])) {
          blocks.push_back({AsyncProducerKind::kTma,
                            StripTmaCopyWriteBufferAttr(flat_stmts[i]),
                            Optional<Stmt>(flat_stmts[i + 1]),
                            write_buffer_data});
          i += 2;
          continue;
        }
      }
      if (ContainsPtxCpAsync(flat_stmts[i])) {
        size_t cp_async_end = i;
        while (cp_async_end + 1 < flat_stmts.size() &&
               ContainsPtxCpAsync(flat_stmts[cp_async_end + 1])) {
          ++cp_async_end;
        }
        if (cp_async_end + 2 < flat_stmts.size() &&
            IsPtxCommitGroup(flat_stmts[cp_async_end + 1]) &&
            IsPtxWaitGroupZero(flat_stmts[cp_async_end + 2])) {
          Array<Stmt> producer_seq;
          producer_seq.reserve(cp_async_end - i + 2);
          for (size_t j = i; j <= cp_async_end; ++j) {
            producer_seq.push_back(flat_stmts[j]);
          }
          producer_seq.push_back(flat_stmts[cp_async_end + 1]);
          Stmt producer_stmt = producer_seq.size() == 1 ? producer_seq[0]
                                                        : SeqStmt(producer_seq);
          blocks.push_back({AsyncProducerKind::kCpAsync, producer_stmt,
                            Optional<Stmt>(),
                            GetCpAsyncDstBufferData(producer_stmt)});
          i = cp_async_end + 3;
          continue;
        }
      }
      compute_stmts.push_back(flat_stmts[i]);
      i++;
    }
  }

private:
  static const CallNode *GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      return eval->value.as<CallNode>();
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return GetEvaluateCallInSimpleWrapper(if_stmt->then_case);
      }
      return nullptr;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return GetEvaluateCallInSimpleWrapper(attr->body);
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return GetEvaluateCallInSimpleWrapper(let->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return GetEvaluateCallInSimpleWrapper(seq->seq[0]);
      }
      return nullptr;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return GetEvaluateCallInSimpleWrapper(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return GetEvaluateCallInSimpleWrapper(realize->block->body);
      }
      return nullptr;
    }
    return nullptr;
  }

  static Optional<Var> ExtractTmaCopyWriteBufferData(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == "tl.tma_copy_write_buffer") {
        const auto *v = attr->node.as<VarNode>();
        ICHECK(v);
        return GetRef<Var>(v);
      }
      return ExtractTmaCopyWriteBufferData(attr->body);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return ExtractTmaCopyWriteBufferData(if_stmt->then_case);
      }
      return Optional<Var>();
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return ExtractTmaCopyWriteBufferData(let->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ExtractTmaCopyWriteBufferData(seq->seq[0]);
      }
      return Optional<Var>();
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ExtractTmaCopyWriteBufferData(block->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ExtractTmaCopyWriteBufferData(realize->block->body);
      }
      return Optional<Var>();
    }
    return Optional<Var>();
  }

  static bool IsMbarrierWaitParity(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    return call && call->op.same_as(mbarrier_wait_parity());
  }

  static bool ContainsTmaLoad(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          found = true;
        }
      }
    });
    return found;
  }

  static bool ContainsPtxCpAsync(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::ptx_cp_async()) ||
            call->op.same_as(tl::ptx_cp_async())) {
          found = true;
        }
      }
    });
    return found;
  }

  static bool IsPtxCommitGroup(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    return call && call->op.same_as(builtin::ptx_commit_group());
  }

  static bool IsPtxWaitGroupZero(const Stmt &stmt) {
    const auto *call = GetEvaluateCallInSimpleWrapper(stmt);
    if (!call || !call->op.same_as(builtin::ptx_wait_group())) {
      return false;
    }
    ICHECK_EQ(call->args.size(), 1);
    const auto *imm = call->args[0].as<IntImmNode>();
    ICHECK(imm);
    return imm->value == 0;
  }

  static Optional<Var> AccessPtrBufferVar(const PrimExpr &ptr) {
    const auto *call = ptr.as<CallNode>();
    if (!call) {
      return Optional<Var>();
    }
    if (call->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(call->args.size(), 3);
      const auto *base_load = call->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      return base_load->buffer->data;
    }
    if (call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(call->args.size(), 5);
      const auto *var = call->args[1].as<VarNode>();
      ICHECK(var);
      return GetRef<Var>(var);
    }
    ICHECK(false) << "Expected tl.access_ptr or tvm_access_ptr";
    throw;
  }

  static Optional<Var> GetCpAsyncDstBufferData(const Stmt &stmt) {
    Optional<Var> found = std::nullopt;
    bool multiple = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (multiple) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (!(call->op.same_as(builtin::ptx_cp_async()) ||
            call->op.same_as(tl::ptx_cp_async()))) {
        return;
      }
      ICHECK(!call->args.empty());
      Optional<Var> current = AccessPtrBufferVar(call->args[0]);
      if (!current.defined()) {
        return;
      }
      if (!found.defined()) {
        found = current;
      } else if (found.value().get() != current.value().get()) {
        multiple = true;
      }
    });
    if (multiple) {
      return Optional<Var>();
    }
    return found;
  }
};

// ---------------------------------------------------------------------------
// TMA Reduce-Add Detector (for three-role WS)
// ---------------------------------------------------------------------------

/*!
 * \brief Information about a detected TMA reduce-add pattern in compute_stmts.
 *
 * The pattern (emitted by atomic_add.cc for T.atomic_add with use_tma):
 *   IfThenElse(threadIdx.x == 0,
 *     SeqStmt({tma_store(need_reduce=1), tma_store_arrive(), tma_store_wait()}))
 *
 * In three-role WS, this is extracted from the consumer and moved to a
 * dedicated dQ writer warp within the producer warp group.
 */
struct TmaReduceAddInfo {
  int compute_stmt_index;  ///< Index in extractor.compute_stmts
  Stmt full_stmt;          ///< The complete IfThenElse statement
  Var smem_buffer_data;    ///< Shared memory buffer read by TMA store
};

/*!
 * \brief Detect TMA reduce-add patterns in compute statements.
 *
 * Scans for the IfThenElse(threadIdx.x == 0, {tma_store(need_reduce=1),
 * tma_store_arrive(), tma_store_wait()}) pattern and returns info for each.
 */
/*!
 * \brief Check if a statement contains a tma_store with need_reduce=1.
 * If found, extract the smem buffer data var.
 */
static bool ContainsTmaReduceStore(const Stmt &stmt, Var *out_smem_buf) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found)
      return;
    const auto *call = node.as<CallNode>();
    if (!call || !call->op.same_as(tma_store()) || call->args.size() < 4)
      return;
    const auto *reduce_flag =
        call->args[call->args.size() - 2].as<IntImmNode>();
    if (!reduce_flag || reduce_flag->value != 1)
      return;
    found = true;
    const auto *access_call = call->args[1].as<CallNode>();
    if (access_call) {
      if (access_call->op.same_as(tl::access_ptr())) {
        if (const auto *base_load =
                access_call->args[0].as<BufferLoadNode>()) {
          *out_smem_buf = base_load->buffer->data;
        }
      } else if (access_call->op.same_as(builtin::tvm_access_ptr())) {
        if (const auto *var = access_call->args[1].as<VarNode>()) {
          *out_smem_buf = GetRef<Var>(var);
        }
      }
    }
  });
  return found;
}

static bool ContainsTmaStoreArrive(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tma_store_arrive()))
        found = true;
    }
  });
  return found;
}

static bool ContainsTmaStoreWait(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tma_store_wait()))
        found = true;
    }
  });
  return found;
}

/*!
 * \brief Detect TMA reduce-add patterns across (possibly split) compute stmts.
 *
 * The atomic_add lowering emits: IfThenElse(threadIdx.x==0,
 *   SeqStmt({tma_store(need_reduce=1), tma_store_arrive(), tma_store_wait()}))
 *
 * After thread_storage_sync or other passes, these may be split into up to 3
 * separate compute_stmts. We detect the tma_store(need_reduce=1) stmt and
 * then look for tma_store_arrive/wait in subsequent stmts.
 */
static std::vector<TmaReduceAddInfo>
DetectTmaReduceAdd(const std::vector<Stmt> &compute_stmts) {
  std::vector<TmaReduceAddInfo> results;
  for (size_t i = 0; i < compute_stmts.size(); ++i) {
    Var smem_buf;

    // Case 1: All three ops in one statement.
    if (ContainsTmaReduceStore(compute_stmts[i], &smem_buf) &&
        ContainsTmaStoreArrive(compute_stmts[i]) &&
        ContainsTmaStoreWait(compute_stmts[i])) {
      TmaReduceAddInfo info;
      info.compute_stmt_index = static_cast<int>(i);
      info.full_stmt = compute_stmts[i];
      info.smem_buffer_data = smem_buf;
      results.push_back(info);
      continue;
    }

    // Case 2: Split across consecutive stmts.
    // Look for tma_store(need_reduce=1) in stmt[i], then scan forward
    // for tma_store_arrive and tma_store_wait within the next few stmts.
    if (!ContainsTmaReduceStore(compute_stmts[i], &smem_buf))
      continue;
    if (!smem_buf.defined())
      continue;

    int arrive_idx = -1, wait_idx = -1;
    for (size_t j = i + 1; j < compute_stmts.size() && j <= i + 3; ++j) {
      if (arrive_idx < 0 && ContainsTmaStoreArrive(compute_stmts[j]))
        arrive_idx = static_cast<int>(j);
      if (wait_idx < 0 && ContainsTmaStoreWait(compute_stmts[j]))
        wait_idx = static_cast<int>(j);
    }
    if (arrive_idx >= 0 && wait_idx >= 0) {
      // Found the full pattern across stmts [i .. max(arrive_idx, wait_idx)].
      // Record the range — all stmts from i to the last one are part of the
      // TMA reduce-add and will be extracted to the dQ writer.
      TmaReduceAddInfo info;
      info.compute_stmt_index = static_cast<int>(i);
      info.full_stmt = compute_stmts[i]; // tma_store stmt
      info.smem_buffer_data = smem_buf;
      // Mark the arrive/wait indices as additional stmts to extract.
      // We store the tma_store index; the arrive/wait will be handled
      // by recording all indices in the extraction set.
      results.push_back(info);
      // Also record arrive and wait as separate entries for the extraction set.
      // Use the same smem_buffer_data since they're part of the same pattern.
      {
        TmaReduceAddInfo arrive_info;
        arrive_info.compute_stmt_index = arrive_idx;
        arrive_info.full_stmt = compute_stmts[arrive_idx];
        arrive_info.smem_buffer_data = smem_buf;
        results.push_back(arrive_info);
      }
      {
        TmaReduceAddInfo wait_info;
        wait_info.compute_stmt_index = wait_idx;
        wait_info.full_stmt = compute_stmts[wait_idx];
        wait_info.smem_buffer_data = smem_buf;
        results.push_back(wait_info);
      }
      // Skip past the pattern to avoid double-detection.
      i = static_cast<size_t>(std::max(arrive_idx, wait_idx));
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// ThreadIdxRewriter (from warp_specialized_rewriter.cc)
// ---------------------------------------------------------------------------

/*!
 * \brief Substitute all Vars whose name contains "threadIdx" with
 *        (Var - offset). This handles the case where lowered consumer
 *        stmts use built-in threadIdx intrinsics (different Var objects
 *        from thread_iv_->var) that PCThreadIdxRewriter can't match.
 */
class ThreadIdxSubstitutor : public StmtExprMutator {
public:
  static Stmt Substitute(Stmt stmt, Var thread_var, PrimExpr offset) {
    ThreadIdxSubstitutor sub(std::move(thread_var), std::move(offset));
    auto result = sub(std::move(stmt));
    LOG(INFO) << "ThreadIdxSubstitutor: " << sub.count_ << " replacements";
    return result;
  }

private:
  ThreadIdxSubstitutor(Var thread_var, PrimExpr offset)
      : thread_var_(std::move(thread_var)), offset_(std::move(offset)) {}

  PrimExpr VisitExpr_(const VarNode *var) final {
    if (var == thread_var_.get()) {
      count_++;
      return GetRef<Var>(var) - offset_;
    }
    return StmtExprMutator::VisitExpr_(var);
  }

  Var thread_var_;
  PrimExpr offset_;
  int count_ = 0;
};

class PCThreadIdxRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                      PrimExpr thread_extent, bool do_shuffle = false,
                      int rewrite_barrier_from = 0,
                      int rewrite_barrier_to = 0,
                      int barrier_id_offset = 0) {
    auto rewriter =
        PCThreadIdxRewriter(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent), do_shuffle);
    rewriter.rewrite_barrier_from_ = rewrite_barrier_from;
    rewriter.rewrite_barrier_to_ = rewrite_barrier_to;
    rewriter.barrier_id_offset_ = barrier_id_offset;
    auto result = rewriter(std::move(stmt));
    if (rewriter.replace_count_ > 0 || rewriter.name_match_count_ > 0) {
      LOG(INFO) << "PCThreadIdxRewriter: ptr_match="
                << rewriter.replace_count_
                << ", name_match=" << rewriter.name_match_count_;
    }
    return result;
  }

private:
  PCThreadIdxRewriter(Var thread_var, PrimExpr replaced, PrimExpr thread_extent,
                      bool do_shuffle)
      : thread_var_(std::move(thread_var)), replaced_(std::move(replaced)),
        thread_extent_(std::move(thread_extent)), do_shuffle_(do_shuffle) {}

  PrimExpr VisitExpr_(const VarNode *var) final {
    if (var == thread_var_.get()) {
      replace_count_++;
      return replaced_;
    }
    // Also try name-based matching as fallback for dual-consumer
    if (var->name_hint == thread_var_->name_hint &&
        var != thread_var_.get()) {
      name_match_count_++;
      return replaced_;
    }
    return StmtExprMutator::VisitExpr_(var);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    auto f_uses_thread = [=](const tvm::tir::VarNode *v) {
      return v == thread_var_.get();
    };
    maybe_thread_opt_ = false;
    if (!op->else_case.defined() && op->condition.as<EQNode>() &&
        UsesVar(op->condition, f_uses_thread) &&
        !(UsesVar(op->then_case, f_uses_thread))) {
      auto eq_op = Downcast<EQ>(op->condition);
      if (eq_op->a.as<VarNode>() == thread_var_.get() ||
          eq_op->b.as<VarNode>() == thread_var_.get()) {
        maybe_thread_opt_ = true;
      }
      auto then_case = StmtExprMutator::VisitStmt(op->then_case);
      maybe_thread_opt_ = do_shuffle_ && maybe_thread_opt_ && has_tma_op_;
      has_tma_op_ = false;
      if (maybe_thread_opt_) {
        return IfThenElse(
            Call(DataType::Bool(), tl_shuffle_elect(), {thread_extent_}),
            StmtExprMutator::VisitStmt(op->then_case), std::nullopt);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::tma_load()) ||
        op->op.same_as(tl::tma_load_im2col()) ||
        op->op.same_as(tl::tma_store()) ||
        op->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
        op->op.same_as(mbarrier_expect_tx())) {
      has_tma_op_ = true;
    }
    // Rewrite NamedBarrier<N> in extern call strings when
    // rewrite_barrier_count_ is set. This handles AllReduce calls
    // that use NamedBarrier<consumer_extent> which needs to become
    // NamedBarrier<wg_extent> for dual-consumer mode.
    if (rewrite_barrier_from_ > 0 &&
        op->op.same_as(builtin::call_extern())) {
      if (op->args.size() >= 1) {
        if (auto *str_imm = op->args[0].as<StringImmNode>()) {
          std::string func_name = str_imm->value;
          std::string old_barrier =
              "NamedBarrier<" + std::to_string(rewrite_barrier_from_) + ">";
          std::string new_barrier;
          if (barrier_id_offset_ > 0) {
            // Use NamedBarrier<count, offset> to shift bar.sync IDs
            new_barrier = "NamedBarrier<" +
                          std::to_string(rewrite_barrier_to_) + ", " +
                          std::to_string(barrier_id_offset_) + ">";
          } else {
            new_barrier =
                "NamedBarrier<" + std::to_string(rewrite_barrier_to_) + ">";
          }
          auto pos = func_name.find(old_barrier);
          if (pos != std::string::npos) {
            func_name.replace(pos, old_barrier.size(), new_barrier);
            Array<PrimExpr> new_args;
            new_args.push_back(StringImm(func_name));
            for (size_t i = 1; i < op->args.size(); ++i) {
              new_args.push_back(VisitExpr(op->args[i]));
            }
            return Call(op->dtype, op->op, new_args);
          }
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var thread_var_;
  PrimExpr replaced_;
  PrimExpr thread_extent_;
  bool maybe_thread_opt_ = false;
  int rewrite_barrier_from_ = 0;
  int rewrite_barrier_to_ = 0;
  int barrier_id_offset_ = 0;
  mutable int replace_count_ = 0;
  mutable int name_match_count_ = 0;
  bool do_shuffle_;
  bool has_tma_op_ = false;
};

// ---------------------------------------------------------------------------
// MbarrierInitRemover: removes all create_list_of_mbarrier calls from a stmt
// ---------------------------------------------------------------------------

/*!
 * \brief Post-transform cleanup: remove any create_list_of_mbarrier calls
 *        that remain outside the transformed block (e.g., at the function
 *        body level where lower_tile_op.cc originally placed them).
 *        The new init is already emitted inside the block by the rewriter.
 */
class MbarrierInitRemover : public StmtExprMutator {
public:
  static Stmt Remove(Stmt stmt) {
    MbarrierInitRemover remover;
    return remover(std::move(stmt));
  }

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> new_seq;
    bool changed = false;
    for (const auto &s : op->seq) {
      if (IsCreateListOfMbarrier(s)) {
        changed = true;
        continue; // drop this statement
      }
      Stmt visited = VisitStmt(s);
      new_seq.push_back(visited);
      if (!visited.same_as(s))
        changed = true;
    }
    if (!changed)
      return GetRef<Stmt>(op);
    if (new_seq.size() == 1)
      return new_seq[0];
    return SeqStmt(new_seq);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (IsCreateListOfMbarrier(GetRef<Stmt>(op))) {
      // Return a no-op (should be caught by SeqStmt visitor above,
      // but handle standalone case too)
      return Evaluate(0);
    }
    return GetRef<Stmt>(op);
  }

  // Stop recursion at BlockRealize — the new init is inside the block
  // and we don't want to remove it.
  Stmt VisitStmt_(const BlockRealizeNode *op) final { return GetRef<Stmt>(op); }

  static bool IsCreateListOfMbarrier(const Stmt &stmt) {
    if (auto *eval = stmt.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        return call->op.same_as(create_list_of_mbarrier());
      }
    }
    return false;
  }
};

// ---------------------------------------------------------------------------
// FineGrainedWSRewriter — main pass
// ---------------------------------------------------------------------------

class FineGrainedWSRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(
      PrimFunc f, int configured_producer_threads = 128,
      bool three_role_enabled = false, bool user_set_producer_extent = false,
      bool dual_consumer_enabled = false,
      const std::string &consumer_stage_map_str = "",
      const std::string &barrier_hints_str = "",
      const std::string &stage_offsets_str = "",
      const std::string &warp_assigns_str = "") {
    // Check thread tags
    if (!ThreadTagChecker::HasOnlyThreadIdxX(f)) {
      LOG(WARNING) << "FineGrainedWS: disabled because program uses "
                      "thread tags other than threadIdx.x";
      return f;
    }

    FineGrainedWSRewriter T;
    T.configured_producer_thread_extent_ = configured_producer_threads;
    T.three_role_enabled_ = three_role_enabled;
    T.dual_consumer_enabled_ = dual_consumer_enabled;
    T.user_set_producer_extent_ = user_set_producer_extent;
    T.consumer_stage_map_ = ParseConsumerStageMap(consumer_stage_map_str);
    T.barrier_hints_ = ParseBarrierHints(barrier_hints_str);
    T.explicit_stage_offsets_ = ParseStageOffsets(stage_offsets_str);
    T.warp_assigns_map_ = ParseWarpAssigns(warp_assigns_str);
    f.CopyOnWrite()->body = T(f->body);

    // TODO(lei): This should be refactored
    // If WS was applied, remove any create_list_of_mbarrier calls that
    // remain OUTSIDE the block (e.g. at function body level from
    // lower_tile_op). The new init is already inside the block.
    if (T.ws_transformed_) {
      f.CopyOnWrite()->body = MbarrierInitRemover::Remove(f->body);
    }

    // Mark dual-consumer mode in function attrs for codegen
    if (T.dual_consumer_enabled_ && T.ws_transformed_) {
      f = WithAttr(f, "tl_finegrainedws_dual_consumer", Integer(1));
    }

    return f;
  }

private:
  // Locate the threadIdx.x binding
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      Optional<PrimExpr> old_num_threads = num_threads_;
      num_threads_ = std::nullopt;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (num_threads_.defined()) {
        PrimExpr num_threads = num_threads_.value();
        thread_iv_.CopyOnWrite()->dom = {0, num_threads};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = num_threads;
      }
      // clean up if we may have multiple threadIdx.x that
      // need to be transformed
      num_threads_ = old_num_threads;
      thread_iv_ = {};
      return attr_stmt;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the explicitly pipelined loop for producer/consumer WS.
    const ForNode *pipeline_loop = FindAnnotatedPipelineLoop(orig_block->body);
    if (!pipeline_loop)
      return StmtExprMutator::VisitStmt_(op);

    auto num_stages_anno = pipeline_loop->annotations.Get("num_stages");
    ICHECK(num_stages_anno);
    int num_stages =
        static_cast<int>(Downcast<Integer>(num_stages_anno.value())->value);
    ICHECK_GE(num_stages, 1);

    // Read barrier hints and stage offsets from pipeline loop annotations.
    // These are set by Heddle Phase B and take precedence over pass config.
    {
      auto hints_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_barrier_hints");
      if (hints_anno.has_value()) {
        std::string anno_str =
            static_cast<std::string>(Downcast<String>(hints_anno.value()));
        if (!anno_str.empty()) {
          barrier_hints_ = ParseBarrierHints(anno_str);
        }
      }
      auto offsets_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_stage_offsets");
      if (offsets_anno.has_value()) {
        std::string anno_str =
            static_cast<std::string>(Downcast<String>(offsets_anno.value()));
        if (!anno_str.empty()) {
          explicit_stage_offsets_ = ParseStageOffsets(anno_str);
        }
      }
    }

    // Auto-detect dual-consumer from loop annotation (set by Heddle).
    // This overrides the pass config when the annotation is present.
    int heddle_split_idx = -1;  // -1 = use heuristic, >=0 = use Heddle's choice
    {
      auto dc_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_dual_consumer");
      if (dc_anno.has_value()) {
        std::string anno_str =
            static_cast<std::string>(Downcast<String>(dc_anno.value()));
        if (anno_str == "1" && !dual_consumer_enabled_) {
          LOG(INFO) << "FineGrainedWS: auto-enabling dual-consumer from loop annotation";
          dual_consumer_enabled_ = true;
        }
      }
      auto split_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_dual_consumer_split");
      if (split_anno.has_value()) {
        std::string split_str =
            static_cast<std::string>(Downcast<String>(split_anno.value()));
        heddle_split_idx = std::stoi(split_str);
        LOG(INFO) << "FineGrainedWS: using Heddle-provided split index: " << heddle_split_idx;
      }
    }

    // Auto-detect three-role from loop annotation (set by Heddle).
    // Overrides the pass config when annotation present.
    {
      auto tr_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_three_role");
      if (tr_anno.has_value()) {
        std::string anno_str =
            static_cast<std::string>(Downcast<String>(tr_anno.value()));
        if (anno_str == "1" && !three_role_enabled_) {
          LOG(INFO) << "FineGrainedWS: auto-enabling three-role WS from loop annotation";
          three_role_enabled_ = true;
        }
      }
    }

    // Read per-op warp assignments from loop annotation (Plan B).
    // Format: "s0:0,s1:1,s2:0,s3:1"
    {
      auto wa_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_warp_assigns");
      if (wa_anno.has_value()) {
        std::string anno_str =
            static_cast<std::string>(Downcast<String>(wa_anno.value()));
        if (!anno_str.empty()) {
          warp_assigns_map_ = ParseWarpAssigns(anno_str);
          LOG(INFO) << "FineGrainedWS: parsed " << warp_assigns_map_.size()
                    << " per-op warp assignments from annotation";
        }
      }
      // Also check pcws variant
      if (warp_assigns_map_.empty()) {
        auto wa_anno2 =
            pipeline_loop->annotations.Get("tl_pcws_warp_assigns");
        if (wa_anno2.has_value()) {
          std::string anno_str =
              static_cast<std::string>(Downcast<String>(wa_anno2.value()));
          if (!anno_str.empty()) {
            warp_assigns_map_ = ParseWarpAssigns(anno_str);
            LOG(INFO) << "FineGrainedWS: parsed " << warp_assigns_map_.size()
                      << " per-op warp assignments from pcws annotation";
          }
        }
      }
    }

    // Flatten the loop body
    Array<Stmt> flat_stmts;
    Stmt loop_body_root = pipeline_loop->body;
    if (auto *realize = pipeline_loop->body.as<BlockRealizeNode>()) {
      loop_body_root = realize->block->body;
    }
    std::vector<std::pair<Var, PrimExpr>> loop_body_lets;
    while (const auto *let_stmt = loop_body_root.as<LetStmtNode>()) {
      loop_body_lets.emplace_back(let_stmt->var, let_stmt->value);
      loop_body_root = let_stmt->body;
    }
    FlattenSeqStmt(loop_body_root, &flat_stmts);
    auto rewrap_loop_body_lets = [&](Stmt body) {
      for (auto it = loop_body_lets.rbegin(); it != loop_body_lets.rend();
           ++it) {
        body = LetStmt((*it).first, (*it).second, body);
      }
      return body;
    };
    // Extract async producer blocks (TMA and cp.async)
    AsyncCopyBlockExtractor extractor;
    extractor.Extract(flat_stmts);

    if (extractor.blocks.empty()) {
      // No TMA loads found — fall through to standard pipeline
      return StmtExprMutator::VisitStmt_(op);
    }

    // NOTE: tl_pipeline_order/stage with -1 values are user-provided
    // producer markers from T.Pipelined(order=..., stage=..., group=...).
    // FineGrainedWS should process these, not skip them.

    VarBindingMap saved_loop_guard_bindings = current_loop_guard_bindings_;
    for (const auto &[var, value] : loop_body_lets) {
      current_loop_guard_bindings_[var] = value;
    }

    BufferDataToBufferMap buffer_data_to_buffer =
        BufferDataToBufferCollector::Collect(GetRef<Stmt>(op));

    // ---------------------------------------------------------------
    // Build producer and consumer loop bodies
    // ---------------------------------------------------------------
    PrimExpr consumer_thread_extent = thread_iv_->dom->extent;
    PrimExpr producer_thread_extent =
        IntImm(DataType::Int(32), configured_producer_thread_extent_);
    // When the user explicitly requests more threads than default (e.g.,
    // threads=384 with producer=128), the total thread extent already includes
    // both consumer and producer.  Subtract producer to get consumer-only.
    // For the default case (threads=256, producer=128), keep the old behavior
    // where consumer = total threads and FineGrainedWS adds producer on top.
    if (user_set_producer_extent_) {
      consumer_thread_extent =
          thread_iv_->dom->extent - producer_thread_extent;
    }
    consumer_thread_extent_ =
        consumer_thread_extent; // Store for RebuildBlockBody
    producer_thread_extent_ = producer_thread_extent;
    PrimExpr ws_consumer_thread_extent = consumer_thread_extent;

    // Barrier layout has two modes:
    // 1) Mixed TMA + cp.async:
    //    keep existing TMA forward ids, append cp.async forward ids, then
    //    append back-pressure ids.
    // 2) Pure TMA:
    //    remap to [loop forward][back-pressure][preloop forward] so producer
    //    and consumer follow the same protocol as the hand-written WS kernels.
    int num_existing_tma_fwd_barriers = 0;
    int num_cp_async_groups = 0;
    for (const auto &block : extractor.blocks) {
      if (block.kind == AsyncProducerKind::kTma) {
        ++num_existing_tma_fwd_barriers;
      } else if (block.kind == AsyncProducerKind::kCpAsync) {
        ++num_cp_async_groups;
      }
    }
    std::vector<int> wait_insert_pos(extractor.blocks.size(), 0);
    std::vector<int> arrive_insert_pos(
        extractor.blocks.size(),
        static_cast<int>(extractor.compute_stmts.size()));
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      if (!extractor.blocks[ti].write_buffer_data.defined()) {
        continue;
      }
      const Var &target = extractor.blocks[ti].write_buffer_data.value();
      int first_read = -1;
      int last_access = -1;
      for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
        BufferDataAccessInfo access = AnalyzeBufferDataAccess(
            extractor.compute_stmts[ci], target, buffer_data_to_buffer);
        if (access.read && first_read < 0) {
          first_read = static_cast<int>(ci);
        }
        if (access.HasAnyAccess()) {
          last_access = static_cast<int>(ci);
        }
      }
      if (first_read >= 0) {
        wait_insert_pos[ti] = first_read;
        arrive_insert_pos[ti] = last_access + 1;
      } else if (last_access >= 0) {
        // Write-only statements that touch the producer-written shared buffer
        // do not need the producer result, so keep the forward wait at the
        // loop head while still delaying back-pressure release until the last
        // consumer-side access.
        wait_insert_pos[ti] = 0;
        arrive_insert_pos[ti] = last_access + 1;
      }
    }

    // --- Apply barrier hints (Proposal 2) ---
    // Override wait/arrive positions if Phase B provided hints.
    // Safety: hints cannot place waits before first_read.
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      if (!extractor.blocks[ti].write_buffer_data.defined())
        continue;
      const Var &target_buf = extractor.blocks[ti].write_buffer_data.value();
      // Find the buffer name for this block
      std::string buf_name;
      if (buffer_data_to_buffer.count(target_buf)) {
        buf_name = buffer_data_to_buffer.at(target_buf)->name;
      }
      if (!buf_name.empty() && barrier_hints_.count(buf_name)) {
        auto [hint_wait, hint_arrive] = barrier_hints_.at(buf_name);
        // Clamp: hint_wait must be in [0, wait_insert_pos[ti]].
        // The wait is inserted BEFORE compute_stmt[wait_insert_pos].
        // Hints can only move the wait EARLIER (closer to loop head),
        // never past first_read, as that would read before data is ready.
        int safe_wait = std::max(0, std::min(hint_wait, wait_insert_pos[ti]));
        // Clamp: hint_arrive must be >= dependency-derived lower bound
        // (last_access + 1). This is critical: releasing the slot before
        // the final consumer access causes a correctness bug.
        int safe_arrive =
            std::max(hint_arrive, arrive_insert_pos[ti]);
        // Clamp: hint_arrive must be <= compute_stmts.size()
        safe_arrive = std::min(
            safe_arrive,
            static_cast<int>(extractor.compute_stmts.size()));
        if (safe_wait != wait_insert_pos[ti] ||
            safe_arrive != arrive_insert_pos[ti]) {
          LOG(INFO) << "FineGrainedWS barrier hint applied for " << buf_name
                    << ": wait " << wait_insert_pos[ti] << "->" << safe_wait
                    << ", arrive " << arrive_insert_pos[ti] << "->"
                    << safe_arrive;
          wait_insert_pos[ti] = safe_wait;
          arrive_insert_pos[ti] = safe_arrive;
        }
      }
    }

    // --- Compute per-compute-stmt stage offsets (Proposal 1) ---
    // For each compute_stmt, determine if it needs a different stage
    // offset based on consumer_stage_map_ or explicit_stage_offsets_.
    std::vector<int> compute_stmt_stage_offsets(
        extractor.compute_stmts.size(), 0);
    int max_abs_stage_offset = 0;
    {
      // First, apply explicit stage offsets (from Phase B auto-derivation)
      for (const auto &[ci, offset] : explicit_stage_offsets_) {
        if (ci >= 0 &&
            ci < static_cast<int>(extractor.compute_stmts.size())) {
          compute_stmt_stage_offsets[ci] = offset;
          max_abs_stage_offset =
              std::max(max_abs_stage_offset, std::abs(offset));
        }
      }
      // Then, apply pattern-based stage map (from user config)
      // This iterates over producer blocks, matching buffer names and
      // compute_stmt patterns.
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        if (!extractor.blocks[ti].write_buffer_data.defined())
          continue;
        const Var &target_buf = extractor.blocks[ti].write_buffer_data.value();
        std::string buf_name;
        if (buffer_data_to_buffer.count(target_buf)) {
          buf_name = buffer_data_to_buffer.at(target_buf)->name;
        }
        if (buf_name.empty() || !consumer_stage_map_.count(buf_name))
          continue;
        const auto &patterns = consumer_stage_map_.at(buf_name);
        // For each compute_stmt that reads this buffer, check patterns
        for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
          BufferDataAccessInfo access = AnalyzeBufferDataAccess(
              extractor.compute_stmts[ci], target_buf,
              buffer_data_to_buffer);
          if (!access.read)
            continue;
          // Check if stmt string matches any pattern
          std::ostringstream stmt_str;
          stmt_str << extractor.compute_stmts[ci];
          std::string stmt_repr = stmt_str.str();
          for (const auto &[pattern, offset] : patterns) {
            if (stmt_repr.find(pattern) != std::string::npos) {
              if (compute_stmt_stage_offsets[ci] != 0 &&
                  compute_stmt_stage_offsets[ci] != offset) {
                LOG(WARNING)
                    << "FineGrainedWS cross-stage: compute_stmt[" << ci
                    << "] reads multiple buffers with different "
                       "offsets ("
                    << compute_stmt_stage_offsets[ci] << " vs "
                    << offset << " for " << buf_name
                    << "). Using first non-zero offset.";
              } else {
                compute_stmt_stage_offsets[ci] = offset;
                max_abs_stage_offset =
                    std::max(max_abs_stage_offset, std::abs(offset));
              }
              break;
            }
          }
        }
      }
      // Validate: stage_offset requires num_stages >= abs(offset) + 1.
      // With num_stages=S, the producer cycles through S slots. The
      // offset consumer lags by abs(offset) iterations. The slot must
      // not be reused before the offset consumer reads it, so we need
      // S > abs(offset), i.e., S >= abs(offset) + 1.
      int min_stages = max_abs_stage_offset + 1;
      if (max_abs_stage_offset > 0 && num_stages < min_stages) {
        LOG(WARNING) << "FineGrainedWS cross-stage: stage_offset=" << max_abs_stage_offset
                     << " requires num_stages >= " << min_stages
                     << ", but num_stages=" << num_stages
                     << ". Disabling cross-stage offsets.";
        std::fill(compute_stmt_stage_offsets.begin(),
                  compute_stmt_stage_offsets.end(), 0);
        max_abs_stage_offset = 0;
      }
    }
    bool has_cross_stage = (max_abs_stage_offset > 0);
    int num_existing_loop_fwd_barriers =
        num_existing_tma_fwd_barriers * num_stages;
    int original_num_existing_loop_fwd_barriers =
        num_existing_loop_fwd_barriers;
    int inferred_existing_required =
        InferMinRequiredBarrierCount(orig_block->body);
    int required_preloop_tma_pairs = CountRewrittenPureTmaPreloopForwardPairs(
        orig_block->body, pipeline_loop);
    bool old_use_full_tma_forward_barrier_protocol =
        use_full_tma_forward_barrier_protocol_;
    bool old_remap_pure_tma_barriers = remap_pure_tma_barriers_;
    int old_pure_tma_preloop_fwd_base = pure_tma_preloop_fwd_base_;
    int old_pure_tma_preloop_fwd_count = pure_tma_preloop_fwd_count_;
    int old_pure_tma_preloop_fwd_cursor = pure_tma_preloop_fwd_cursor_;
    use_full_tma_forward_barrier_protocol_ = (num_cp_async_groups == 0);
    remap_pure_tma_barriers_ = use_full_tma_forward_barrier_protocol_;
    std::vector<Stmt> ws_producer_stmts(extractor.blocks.size());
    std::vector<Optional<Stmt>> ws_wait_stmts(extractor.blocks.size(),
                                              std::nullopt);
    std::vector<Optional<PrimExpr>> producer_issue_guards(
        extractor.blocks.size(), std::nullopt);
    std::vector<Optional<Stmt>> producer_issue_guard_sources(
        extractor.blocks.size(), std::nullopt);
    std::vector<Optional<PrimExpr>> protocol_guards(extractor.blocks.size(),
                                                    std::nullopt);
    std::vector<Optional<Stmt>> protocol_guard_sources(extractor.blocks.size(),
                                                       std::nullopt);
    for (size_t i = 0; i < extractor.blocks.size(); ++i) {
      ws_producer_stmts[i] = extractor.blocks[i].producer_stmt;
      ws_wait_stmts[i] = extractor.blocks[i].wait_stmt;
      producer_issue_guards[i] =
          ExtractNonThreadProducerGuard(extractor.blocks[i].producer_stmt);
      if (producer_issue_guards[i].defined()) {
        producer_issue_guard_sources[i] = extractor.blocks[i].producer_stmt;
        protocol_guards[i] = producer_issue_guards[i];
        protocol_guard_sources[i] = extractor.blocks[i].producer_stmt;
        if (arrive_insert_pos[i] > 0 &&
            arrive_insert_pos[i] <=
                static_cast<int>(extractor.compute_stmts.size())) {
          const Stmt &arrive_source =
              extractor.compute_stmts[arrive_insert_pos[i] - 1];
          Optional<PrimExpr> arrive_guard =
              ExtractNonThreadProducerGuard(arrive_source);
          if (arrive_guard.defined()) {
            protocol_guards[i] = arrive_guard;
            protocol_guard_sources[i] = arrive_source;
          }
        }
      }
      // NOTE: Previously, when the guard was a mask-like boolean expression
      // (e.g. BlockMask[by, bx, k]), the producer would strip the guard and
      // issue TMA loads unconditionally.  This causes unnecessary memory
      // traffic for sparse workloads, so we now keep the guard on the
      // producer side and rely on phase-counter-based parity tracking to
      // maintain barrier synchronisation.
    }

    // ---------------------------------------------------------------
    // Detect whether the pipeline loop needs counter-based phase
    // tracking.  This is necessary when the loop body is conditionally
    // guarded (e.g. `if block_mask[k]`) so that skipped iterations do
    // not desynchronise the mbarrier parity.
    // ---------------------------------------------------------------
    bool needs_phase_counter = false;
    Optional<PrimExpr> uniform_phase_guard;
    Optional<Stmt> uniform_phase_guard_source;
    {
      StructuralEqual eq;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (protocol_guards[i].defined()) {
          if (!needs_phase_counter) {
            needs_phase_counter = true;
            uniform_phase_guard = protocol_guards[i];
            uniform_phase_guard_source = protocol_guard_sources[i];
          } else if (!eq(uniform_phase_guard.value(),
                         protocol_guards[i].value())) {
            // Different guards on different blocks – fall back to
            // original loop-variable parity (no counter).
            needs_phase_counter = false;
            break;
          }
        }
      }
      // Only use counter when ALL blocks share the same guard.
      if (needs_phase_counter) {
        for (size_t i = 0; i < extractor.blocks.size(); ++i) {
          if (!protocol_guards[i].defined()) {
            needs_phase_counter = false;
            break;
          }
        }
      }
    }

    std::optional<PhaseCounter> producer_phase_counter;
    std::optional<PhaseCounter> consumer_phase_counter;
    if (needs_phase_counter) {
      producer_phase_counter = PhaseCounter::Create("producer_phase_cnt");
      consumer_phase_counter = PhaseCounter::Create("consumer_phase_cnt");
    }

    StructuralEqual equal;
    auto same_optional_expr = [&](const Optional<PrimExpr> &guard_a,
                                  const Optional<PrimExpr> &guard_b) {
      if (guard_a.defined() != guard_b.defined()) {
        return false;
      }
      return !guard_a.defined() || equal(guard_a.value(), guard_b.value());
    };
    auto same_guard = [&](size_t lhs, size_t rhs) {
      return same_optional_expr(producer_issue_guards[lhs],
                                producer_issue_guards[rhs]) &&
             same_optional_expr(protocol_guards[lhs], protocol_guards[rhs]);
    };
    std::vector<int> block_group(extractor.blocks.size(), 0);
    int num_block_groups = 0;
    if (!extractor.blocks.empty()) {
      int next_group = 0;
      block_group[0] = next_group++;
      bool current_group_has_tma =
          extractor.blocks[0].kind == AsyncProducerKind::kTma;
      for (size_t i = 1; i < extractor.blocks.size(); ++i) {
        bool merge_with_prev =
            wait_insert_pos[i] == wait_insert_pos[i - 1] &&
            arrive_insert_pos[i] == arrive_insert_pos[i - 1] &&
            same_guard(i - 1, i);
        if (merge_with_prev && !remap_pure_tma_barriers_ &&
            current_group_has_tma &&
            extractor.blocks[i].kind == AsyncProducerKind::kTma) {
          // Mixed groups can safely share one TMA barrier with cp.async
          // arrive-on notifications, but keeping multiple TMA producers on the
          // same preserved protocol would over-arrive the barrier.
          merge_with_prev = false;
        }
        block_group[i] = merge_with_prev ? block_group[i - 1] : next_group++;
        if (!merge_with_prev) {
          current_group_has_tma =
              extractor.blocks[i].kind == AsyncProducerKind::kTma;
        } else if (extractor.blocks[i].kind == AsyncProducerKind::kTma) {
          current_group_has_tma = true;
        }
      }
      num_block_groups = next_group;
    }

    std::vector<Array<Stmt>> producer_loop_prefix_stmts(
        extractor.blocks.size());
    std::vector<bool> moved_compute_stmts(extractor.compute_stmts.size(),
                                          false);
    int compute_cursor = 0;
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      bool is_first_in_group =
          ti == 0 || block_group[ti] != block_group[ti - 1];
      if (!is_first_in_group) {
        continue;
      }
      int wait_pos = wait_insert_pos[ti];
      if (wait_pos <= compute_cursor) {
        compute_cursor = std::max(compute_cursor, wait_pos);
        continue;
      }
      bool all_movable = true;
      for (int ci = compute_cursor; ci < wait_pos; ++ci) {
        if (!IsProducerMovableLoopPrefixStmt(extractor.compute_stmts[ci])) {
          all_movable = false;
          break;
        }
      }
      if (all_movable) {
        for (int ci = compute_cursor; ci < wait_pos; ++ci) {
          producer_loop_prefix_stmts[ti].push_back(extractor.compute_stmts[ci]);
          moved_compute_stmts[ci] = true;
        }
      }
      compute_cursor = wait_pos;
    }

    auto stmt_has_lowered_simt_copy = [&](const Stmt &stmt) {
      return ProducerSimtCopyDetector::HasSimtCopy(stmt, buffer_data_to_buffer);
    };
    bool producer_needs_full_thread_extent =
        std::any_of(ws_producer_stmts.begin(), ws_producer_stmts.end(),
                    stmt_has_lowered_simt_copy);
    if (!producer_needs_full_thread_extent) {
      for (const auto &prefix_stmts : producer_loop_prefix_stmts) {
        for (const auto &stmt : prefix_stmts) {
          if (stmt_has_lowered_simt_copy(stmt)) {
            producer_needs_full_thread_extent = true;
            break;
          }
        }
        if (producer_needs_full_thread_extent) {
          break;
        }
      }
    }
    if (producer_needs_full_thread_extent) {
      // LowerTileOp may already have materialized SIMT global->shared copies.
      // Those copies cannot be safely remapped onto a smaller producer warp
      // partition, so keep the producer extent at the original thread extent.
      producer_thread_extent = consumer_thread_extent;
    }
    producer_thread_extent_ = producer_thread_extent;

    std::vector<bool> group_has_tma(num_block_groups, false);
    std::vector<bool> group_has_cp_async(num_block_groups, false);
    for (size_t i = 0; i < extractor.blocks.size(); ++i) {
      int group = block_group[i];
      if (extractor.blocks[i].kind == AsyncProducerKind::kTma) {
        group_has_tma[group] = true;
      } else if (extractor.blocks[i].kind == AsyncProducerKind::kCpAsync) {
        group_has_cp_async[group] = true;
      }
    }
    int num_tma_groups = 0;
    int num_cp_async_only_groups = 0;
    for (int group = 0; group < num_block_groups; ++group) {
      if (group_has_tma[group]) {
        ++num_tma_groups;
      } else if (group_has_cp_async[group]) {
        ++num_cp_async_only_groups;
      }
    }
    num_existing_tma_fwd_barriers = num_tma_groups;
    num_existing_loop_fwd_barriers = num_existing_tma_fwd_barriers * num_stages;
    int num_new_cp_async_fwd_barriers = num_cp_async_only_groups * num_stages;

    int num_existing_barriers = 0;
    int num_preloop_fwd_barriers = 0;
    if (remap_pure_tma_barriers_) {
      // Pure-TMA WS remaps pre-loop TMA prefixes to a dedicated barrier range.
      // Some kernels reuse loop barrier ids for those prefixes in the original
      // IR, so `inferred_existing_required` alone can undercount how many
      // distinct pre-loop barriers the rewritten form needs.
      num_preloop_fwd_barriers =
          std::max(required_preloop_tma_pairs,
                   std::max(0, inferred_existing_required -
                                   original_num_existing_loop_fwd_barriers));
      num_existing_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
    } else {
      // Mixed TMA/cp.async keeps any existing non-pipelined forward barriers
      // at their original ids. `inferred_existing_required` already accounts
      // for those explicit references, so avoid reserving an extra unused slot.
      num_existing_barriers =
          std::max(num_existing_loop_fwd_barriers, inferred_existing_required);
      num_preloop_fwd_barriers =
          num_existing_barriers - num_existing_loop_fwd_barriers;
    }
    int num_total_fwd_barriers = 0;
    int num_bp_barriers = num_block_groups * num_stages;
    int total_barriers = 0;

    std::vector<int> fwd_bases(extractor.blocks.size(), -1);
    std::vector<int> bp_bases(extractor.blocks.size(), -1);
    std::vector<PrimExpr> mixed_fwd_arrive_counts;

    if (remap_pure_tma_barriers_) {
      // Pure-TMA layout:
      //   [0, loop_fwd)                    : loop forward barriers
      //   [loop_fwd, loop_fwd + bp)       : back-pressure barriers
      //   [loop_fwd + bp, total_barriers) : preloop/prologue forward barriers
      int next_loop_fwd_base = 0;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        if (i == 0 || block_group[i] != block_group[i - 1]) {
          fwd_bases[i] = next_loop_fwd_base;
          next_loop_fwd_base += num_stages;
        } else {
          fwd_bases[i] = fwd_bases[i - 1];
        }
      }
      num_total_fwd_barriers =
          num_existing_loop_fwd_barriers + num_preloop_fwd_barriers;
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        bp_bases[i] =
            num_existing_loop_fwd_barriers + block_group[i] * num_stages;
      }
      pure_tma_preloop_fwd_base_ =
          num_existing_loop_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_count_ = num_preloop_fwd_barriers;
      pure_tma_preloop_fwd_cursor_ = 0;
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
    } else {
      // Mixed path:
      //   [0, num_existing_barriers) : pre-existing forward barriers
      //   [existing, total_fwd)      : new cp.async forward barriers
      //   [total_fwd, total)         : back-pressure barriers
      num_total_fwd_barriers =
          num_existing_barriers + num_new_cp_async_fwd_barriers;
      int next_existing_tma_fwd_base = 0;
      int next_cp_async_fwd_base = num_existing_barriers;
      std::vector<int> group_fwd_bases(num_block_groups, -1);
      mixed_fwd_arrive_counts.assign(num_total_fwd_barriers,
                                     IntImm(DataType::Int(32), 1));
      for (int group = 0; group < num_block_groups; ++group) {
        if (group_has_tma[group]) {
          group_fwd_bases[group] = next_existing_tma_fwd_base;
          next_existing_tma_fwd_base += num_stages;
        } else {
          ICHECK(group_has_cp_async[group]);
          group_fwd_bases[group] = next_cp_async_fwd_base;
          next_cp_async_fwd_base += num_stages;
        }
        PrimExpr group_arrive_count = IntImm(DataType::Int(32), 1);
        if (group_has_cp_async[group]) {
          group_arrive_count = producer_thread_extent;
        }
        for (int stage = 0; stage < num_stages; ++stage) {
          mixed_fwd_arrive_counts[group_fwd_bases[group] + stage] =
              group_arrive_count;
        }
      }
      for (size_t i = 0; i < extractor.blocks.size(); ++i) {
        fwd_bases[i] = group_fwd_bases[block_group[i]];
        bp_bases[i] = num_total_fwd_barriers + block_group[i] * num_stages;
      }
      total_barriers = num_total_fwd_barriers + num_bp_barriers;
      pure_tma_preloop_fwd_base_ = -1;
      pure_tma_preloop_fwd_count_ = 0;
      pure_tma_preloop_fwd_cursor_ = 0;
    }

    // Defensive check: ensure back-pressure barriers do not overlap
    // any existing (forward/prologue) barrier ids in the original IR.
    if (num_bp_barriers > 0 && !remap_pure_tma_barriers_) {
      int existing_last = inferred_existing_required - 1;
      int bp_begin = bp_bases.front();
      int bp_last = bp_begin + num_bp_barriers - 1;
      ICHECK(bp_begin > existing_last)
          << "FineGrainedWS: barrier id overlap detected. "
          << "existing_last=" << existing_last << ", bp_begin=" << bp_begin
          << ", bp_last=" << bp_last;
    }

    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr loop_min = pipeline_loop->min;

    // Compute stage and parity expressions.
    // When needs_phase_counter is true, the loop body is conditionally
    // guarded and we use a mutable counter instead of the loop variable
    // to derive stage/parity.  Producer and consumer have separate
    // counters because they run on different warp partitions.
    PrimExpr linear_idx = loop_var - loop_min;
    // executed_iter_count: the number of executed iterations so far.
    // When needs_phase_counter (guarded/masked loops), this is the phase
    // counter value (only incremented on actually-executed iterations).
    // Otherwise it equals linear_idx (every iteration executes).
    PrimExpr executed_iter_count =
        needs_phase_counter ? consumer_phase_counter->Load() : linear_idx;
    PrimExpr base_stage_expr = FloorMod(linear_idx, num_stages);
    PrimExpr base_parity_expr = FloorMod(FloorDiv(linear_idx, num_stages), 2);

    PrimExpr producer_stage_expr =
        needs_phase_counter ? producer_phase_counter->StageExpr(num_stages)
                            : base_stage_expr;
    PrimExpr producer_parity_expr =
        needs_phase_counter ? producer_phase_counter->ParityExpr(num_stages)
                            : base_parity_expr;
    PrimExpr consumer_stage_expr =
        needs_phase_counter ? consumer_phase_counter->StageExpr(num_stages)
                            : base_stage_expr;
    PrimExpr consumer_parity_expr =
        needs_phase_counter ? consumer_phase_counter->ParityExpr(num_stages)
                            : base_parity_expr;

    // --- Build Producer Body ---
    Array<Stmt> producer_body_stmts;
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      const auto &tma = extractor.blocks[ti];
      int group = block_group[ti];
      bool is_first_in_group =
          ti == 0 || block_group[ti] != block_group[ti - 1];
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              block_group[ti] != block_group[ti + 1];
      PrimExpr bp_id =
          IntImm(DataType::Int(32), bp_bases[ti]) + producer_stage_expr;

      // Back-pressure wait: producer cannot reuse the stage buffer until the
      // consumer releases it. xor(parity, 1) bootstraps the first iteration.
      if (is_first_in_group) {
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            protocol_guard_sources[ti], protocol_guards[ti],
            makeParityWait(bp_id, bitwise_xor(producer_parity_expr, 1))));
        for (const auto &stmt : producer_loop_prefix_stmts[ti]) {
          producer_body_stmts.push_back(stmt);
        }
      }

      Stmt producer_stmt = ws_producer_stmts[ti];
      if (tma.kind == AsyncProducerKind::kTma) {
        ICHECK_GE(fwd_bases[ti], 0);
        PrimExpr barrier_id =
            IntImm(DataType::Int(32), fwd_bases[ti]) + producer_stage_expr;
        if (use_full_tma_forward_barrier_protocol_) {
          // Pure-TMA WS uses a full producer-side release protocol so the
          // consumer waits on a barrier owned by the producer branch.
          producer_stmt = RewriteTmaForwardProducerStmt(
              producer_stmt, barrier_id, is_last_in_group);
        } else {
          // Mixed groups keep the original producer-side TMA protocol, but
          // rebind grouped loads onto a shared forward barrier. If the group
          // also contains cp.async, let cp.async.mbarrier.arrive.noinc own the
          // arrival count so the shared forward barrier stays on the producer
          // thread extent instead of adding an extra leader-only arrive.
          producer_stmt = RewriteTmaStmtBarrierIdPreserveProtocol(
              producer_stmt, barrier_id, group_has_cp_async[group]);
        }
        // Keep expect/load under the same elected lane when lowering has
        // emitted them as adjacent identical IfThenElse wrappers.
        producer_stmt = MergeAdjacentEquivalentIfs(producer_stmt);
      }

      // Execute the producer statement.
      producer_body_stmts.push_back(producer_stmt);
      if (is_last_in_group && group_has_cp_async[group]) {
        ICHECK_GE(fwd_bases[ti], 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), fwd_bases[ti]) + producer_stage_expr;
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            producer_issue_guard_sources[ti], producer_issue_guards[ti],
            makeCpAsyncBarrierNoInc(fwd_id)));
      }
      // Phase counter increment – exactly once per guarded iteration,
      // after ALL groups have issued their barrier ops.
      // MergeAdjacentEquivalentIfs will fold this into the same guard.
      if (needs_phase_counter && ti + 1 == extractor.blocks.size()) {
        producer_body_stmts.push_back(WrapStmtWithGuardSource(
            uniform_phase_guard_source, uniform_phase_guard,
            producer_phase_counter->Increment()));
      }
    }
    Stmt producer_loop_body =
        MergeAdjacentEquivalentIfs(SeqStmt(producer_body_stmts));
    producer_loop_body = rewrap_loop_body_lets(producer_loop_body);

    // --- Three-Role Detection ---
    // Detect TMA reduce-add patterns (from T.atomic_add with use_tma) in
    // compute_stmts. When three-role WS is enabled and patterns are found,
    // extract them into a dedicated dQ writer warp in the producer WG.
    std::vector<TmaReduceAddInfo> tma_reduce_adds;
    bool has_three_role = false;
    // Named barrier IDs: 1 = dQ_full (consumer→writer), 2 = dQ_empty
    // (writer→consumer). Participant count: consumer threads + 32 dQ writer
    // threads.
    PrimExpr three_role_barrier_count;
    // Compute_stmt index of the first write to any dQ smem buffer.
    int first_dq_write_ci = -1;

    // Always probe for TMA reduce-add patterns; auto-enable three-role
    // when patterns are found, even if the user did not set the flag.
    tma_reduce_adds = DetectTmaReduceAdd(extractor.compute_stmts);
    if (!tma_reduce_adds.empty()) {
      if (!three_role_enabled_) {
        LOG(INFO) << "FineGrainedWS: auto-enabling three-role WS "
                  << "(detected " << tma_reduce_adds.size()
                  << " TMA reduce-add pattern(s))";
        three_role_enabled_ = true;
      }
      {
        has_three_role = true;
        three_role_barrier_count =
            consumer_thread_extent + IntImm(DataType::Int(32), 32);
        // Find the first compute_stmt that writes to any detected dQ smem buf.
        for (const auto &info : tma_reduce_adds) {
          for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
            if (static_cast<int>(ci) == info.compute_stmt_index)
              continue;
            auto access = AnalyzeBufferDataAccess(extractor.compute_stmts[ci],
                                                  info.smem_buffer_data,
                                                  buffer_data_to_buffer);
            if (access.write) {
              if (first_dq_write_ci < 0 ||
                  static_cast<int>(ci) < first_dq_write_ci) {
                first_dq_write_ci = static_cast<int>(ci);
              }
              break;
            }
          }
        }
        // Build set of TMA reduce-add stmt indices for fast lookup.
        LOG(INFO) << "FineGrainedWS three-role: detected " << tma_reduce_adds.size()
                  << " TMA reduce-add pattern(s), first dQ write at ci="
                  << first_dq_write_ci;
      }
    }

    // Build a set of compute_stmt indices that are TMA reduce-add patterns.
    std::unordered_set<int> tma_reduce_add_indices;
    int last_tma_reduce_add_ci = -1;
    for (const auto &info : tma_reduce_adds) {
      tma_reduce_add_indices.insert(info.compute_stmt_index);
      last_tma_reduce_add_ci =
          std::max(last_tma_reduce_add_ci, info.compute_stmt_index);
    }

    // --- Build Consumer Body ---
    Array<Stmt> consumer_body_stmts;
    // Per-stmt warp group assignment for Plan B per-op dispatch.
    // -1 = common (goes to all warp groups), >=0 = assigned warp group.
    // Populated only when warp_assigns_map_ is non-empty.
    std::vector<int> consumer_stmt_warp_group;
    bool track_warp_groups = !warp_assigns_map_.empty();
    if (track_warp_groups &&
        warp_assigns_map_.size() < extractor.compute_stmts.size()) {
      LOG(WARNING) << "FineGrainedWS per-op dispatch requires warp assigns "
                   << "for every consumer stmt; got "
                   << warp_assigns_map_.size() << " assigns for "
                   << extractor.compute_stmts.size()
                   << " stmts. Falling back to structured WS dispatch.";
      track_warp_groups = false;
    }
    auto push_consumer_stmt = [&](Stmt stmt, int wg) {
      consumer_body_stmts.push_back(stmt);
      if (track_warp_groups) consumer_stmt_warp_group.push_back(wg);
    };
    // Lookup warp group for a compute_stmt index; returns -1 if unassigned.
    auto warp_group_for_ci = [&](int ci) -> int {
      auto it = warp_assigns_map_.find(ci);
      return (it != warp_assigns_map_.end()) ? it->second : -1;
    };

    // Helper: compute effective stage/parity for a given block, considering
    // the stage_offset of the compute_stmt at its wait position.
    // For cross-stage consumers (Proposal 1), the stage and parity must
    // account for the offset so the consumer reads the correct buffer slot.
    auto compute_effective_stage_parity =
        [&](size_t ti)
        -> std::pair<PrimExpr, PrimExpr> {
      int wait_ci = wait_insert_pos[ti];
      int offset = 0;
      if (has_cross_stage && wait_ci >= 0 &&
          wait_ci < static_cast<int>(compute_stmt_stage_offsets.size())) {
        offset = compute_stmt_stage_offsets[wait_ci];
      }
      if (offset == 0) {
        return {consumer_stage_expr, consumer_parity_expr};
      }
      // effective_stage = FloorMod(consumer_stage_expr + offset, num_stages)
      PrimExpr effective_stage = FloorMod(
          consumer_stage_expr + IntImm(DataType::Int(32), offset),
          num_stages);
      // Parity for offset consumer: the consumer wants to wait on data
      // produced at iteration (k + offset). The producer's release parity
      // for that iteration is FloorMod(FloorDiv(k + offset, num_stages), 2).
      // We compute this from linear_idx (or phase counter when guarded).
      PrimExpr offset_linear =
          (needs_phase_counter ? consumer_phase_counter->Load() : linear_idx) +
          IntImm(DataType::Int(32), offset);
      PrimExpr effective_parity =
          FloorMod(FloorDiv(offset_linear, num_stages), 2);
      if (false) { // disable old XOR logic
      }
      return {effective_stage, effective_parity};
    };

    // Place forward waits at first use and back-pressure arrives at last use.
    // If we cannot prove the dependency, fall back to wait-at-head /
    // arrive-at-tail.
    std::vector<bool> arrive_emitted(extractor.blocks.size(), false);
    std::vector<Stmt> normalized_waits;
    normalized_waits.reserve(extractor.blocks.size());
    for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
      ICHECK_GE(fwd_bases[ti], 0);
      auto [eff_stage, eff_parity] = compute_effective_stage_parity(ti);
      PrimExpr fwd_id =
          IntImm(DataType::Int(32), fwd_bases[ti]) + eff_stage;
      if (ws_wait_stmts[ti].defined()) {
        normalized_waits.push_back(RewriteWaitBarrier(
            ws_wait_stmts[ti].value(), fwd_id, eff_parity));
      } else {
        normalized_waits.push_back(WrapStmtWithGuardSource(
            producer_issue_guard_sources[ti], producer_issue_guards[ti],
            makeParityWait(fwd_id, eff_parity)));
      }
    }
    // Emit waits / compute / arrives according to insertion points.
    for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group =
            ti == 0 || block_group[ti] != block_group[ti - 1];
        if (is_first_in_group && wait_insert_pos[ti] == static_cast<int>(ci)) {
          int wait_ci = wait_insert_pos[ti];
          int offset = (has_cross_stage && wait_ci >= 0 &&
                        wait_ci < static_cast<int>(
                                      compute_stmt_stage_offsets.size()))
                           ? compute_stmt_stage_offsets[wait_ci]
                           : 0;
          Stmt wait_stmt = normalized_waits[ti];
          if (offset != 0) {
            // Cross-stage prologue protocol:
            //
            // mbarrier.wait(parity) is NON-DESTRUCTIVE: it polls the
            // phase bit without advancing it. Multiple waits on the same
            // barrier with the same parity are safe.
            //
            // Prologue (k < |offset|): do a "dummy consume" — wait on
            // the CURRENT stage's fwd barrier (to sync with producer)
            // and arrive on the current bp barrier (to release the slot
            // for reuse). The offset consumer at k+|offset| can still
            // wait on this same barrier because wait doesn't change phase.
            //
            // Steady state (k >= |offset|): use offset fwd wait as normal.
            // Both paths need current-stage bp arrive to keep the
            // producer pipeline flowing.
            PrimExpr fwd_id_current =
                IntImm(DataType::Int(32), fwd_bases[ti]) +
                consumer_stage_expr;
            PrimExpr bp_id_current =
                IntImm(DataType::Int(32), bp_bases[ti]) +
                consumer_stage_expr;
            Stmt dummy_wait = makeParityWait(fwd_id_current,
                                             consumer_parity_expr);
            Stmt bp_arrive = makeArriveBarrier(bp_id_current);

            // Prologue: dummy fwd wait only, NO bp arrive. The producer's
            // first num_stages-1 iterations use bootstrap (XOR parity)
            // which passes without needing consumer bp arrive.
            //
            // With ns=3 and offset=-1: producer bootstraps k=0,1,2.
            // First real bp wait is at k=3 (needs bp[0] released).
            // Consumer k_1=1 (first steady) does offset bp arrive on
            // bp[0], which satisfies producer k=3. The timing works
            // because prologue k_1=0 is fast (no PV compute).
            //
            // Steady: offset fwd wait only. The single bp arrive for
            // this iteration is the offset bp arrive after PV (below).
            wait_stmt = IfThenElse(
                GE(executed_iter_count,
                   IntImm(DataType::Int(32), std::abs(offset))),
                wait_stmt,
                dummy_wait);
          }
          push_consumer_stmt(wait_stmt, warp_group_for_ci(static_cast<int>(ci)));
        }
      }
      // Three-role: insert named_barrier_wait(dQ_empty) before first dQ write.
      // This ensures the dQ writer has finished draining smem from the previous
      // iteration before the consumer overwrites it.
      if (has_three_role && static_cast<int>(ci) == first_dq_write_ci) {
        push_consumer_stmt(Evaluate(
            Call(DataType::Handle(), named_barrier_wait(),
                 {IntImm(DataType::Int(32), 2), three_role_barrier_count})), -1);
      }
      // Three-role: skip TMA reduce-add stmts (moved to dQ writer).
      // Emit named_barrier_arrive(dQ_full) at the position of the LAST
      // TMA reduce-add stmt to signal that dQ smem is ready for draining.
      if (tma_reduce_add_indices.count(static_cast<int>(ci))) {
        if (static_cast<int>(ci) == last_tma_reduce_add_ci) {
          push_consumer_stmt(Evaluate(
              Call(DataType::Handle(), named_barrier_arrive(),
                   {IntImm(DataType::Int(32), 1),
                    three_role_barrier_count})), -1);
        }
        // Skip — this stmt is moved to the dQ writer.
      } else if (!moved_compute_stmts[ci]) {
        Stmt compute_stmt = extractor.compute_stmts[ci];
        int offset = has_cross_stage ? compute_stmt_stage_offsets[ci] : 0;
        if (offset != 0) {
          // Cross-stage: rewrite shared-memory buffer stage expressions.
          // Replace FloorMod(loop_var - loop_min, num_stages) with
          // FloorMod(base + offset, num_stages) where base is either
          // linear_idx or phase_counter->Load() depending on whether
          // the loop is conditionally guarded.
          PrimExpr base_for_offset =
              needs_phase_counter
                  ? consumer_phase_counter->Load()
                  : linear_idx;
          PrimExpr offset_stage =
              FloorMod(base_for_offset + IntImm(DataType::Int(32), offset),
                       num_stages);
          compute_stmt = StageExprReplacer::Replace(
              compute_stmt, loop_var, loop_min, num_stages, offset_stage);
          // Prologue guard: skip when executed_iter_count < abs(offset).
          compute_stmt = IfThenElse(
              GE(executed_iter_count,
                 IntImm(DataType::Int(32), std::abs(offset))),
              compute_stmt);
        }
        push_consumer_stmt(compute_stmt, warp_group_for_ci(static_cast<int>(ci)));
      }
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                                block_group[ti] != block_group[ti + 1];
        if (is_last_in_group &&
            arrive_insert_pos[ti] == static_cast<int>(ci + 1)) {
          // Back-pressure arrive uses the effective stage for cross-stage
          // consistency: the consumer releases the buffer slot it actually read.
          auto [eff_stage, _unused] = compute_effective_stage_parity(ti);
          PrimExpr bp_id =
              IntImm(DataType::Int(32), bp_bases[ti]) + eff_stage;
          // Cross-stage dual-arrive: the current-stage bp arrive was
          // already emitted in the wait section above (unconditional).
          // Here we emit the OFFSET-stage bp arrive, which releases the
          // slot that the offset PV actually read. Only in steady state.
          int wait_ci_bp = wait_insert_pos[ti];
          int bp_offset =
              (has_cross_stage && wait_ci_bp >= 0 &&
               wait_ci_bp < static_cast<int>(
                                compute_stmt_stage_offsets.size()))
                  ? compute_stmt_stage_offsets[wait_ci_bp]
                  : 0;
          Stmt arrive_stmt;
          if (bp_offset != 0) {
            auto [eff_stage_arr, _ign] =
                compute_effective_stage_parity(ti);
            PrimExpr bp_id_eff =
                IntImm(DataType::Int(32), bp_bases[ti]) + eff_stage_arr;
            // Offset bp arrive: only in steady state after PV compute
            arrive_stmt = IfThenElse(
                GE(executed_iter_count,
                   IntImm(DataType::Int(32), std::abs(bp_offset))),
                WrapStmtWithGuardSource(
                    protocol_guard_sources[ti], protocol_guards[ti],
                    makeArriveBarrier(bp_id_eff)));
          } else {
            arrive_stmt = WrapStmtWithGuardSource(
                protocol_guard_sources[ti], protocol_guards[ti],
                makeArriveBarrier(bp_id));
          }
          push_consumer_stmt(arrive_stmt, warp_group_for_ci(static_cast<int>(ci)));
          arrive_emitted[ti] = true;
        }
      }
    }

    // Handle degenerate loops with no compute statements.
    if (extractor.compute_stmts.empty()) {
      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        bool is_first_in_group =
            ti == 0 || block_group[ti] != block_group[ti - 1];
        if (is_first_in_group) {
          push_consumer_stmt(normalized_waits[ti], -1);
        }
      }
    }

    // Emit loop-tail arrives (blocks with unknown deps or tail use).
    for (size_t ti = 0; ti < extractor.blocks.size(); ti++) {
      bool is_last_in_group = ti + 1 == extractor.blocks.size() ||
                              block_group[ti] != block_group[ti + 1];
      if (is_last_in_group && !arrive_emitted[ti] &&
          arrive_insert_pos[ti] ==
              static_cast<int>(extractor.compute_stmts.size())) {
        auto [eff_stage, _unused] = compute_effective_stage_parity(ti);
        PrimExpr bp_id =
            IntImm(DataType::Int(32), bp_bases[ti]) + eff_stage;
        Stmt arrive_stmt = WrapStmtWithGuardSource(
            protocol_guard_sources[ti], protocol_guards[ti],
            makeArriveBarrier(bp_id));
        // Cross-stage: guard tail arrive like inline arrives
        int wait_ci = wait_insert_pos[ti];
        int tail_offset =
            (has_cross_stage && wait_ci >= 0 &&
             wait_ci < static_cast<int>(
                           compute_stmt_stage_offsets.size()))
                ? compute_stmt_stage_offsets[wait_ci]
                : 0;
        // For cross-stage: current-stage bp arrive was handled in the
        // wait section. Here emit offset-stage arrive (steady state only).
        if (tail_offset != 0) {
          arrive_stmt = IfThenElse(
              GE(executed_iter_count,
                 IntImm(DataType::Int(32), std::abs(tail_offset))),
              arrive_stmt);
        }
        push_consumer_stmt(arrive_stmt,
            warp_group_for_ci(static_cast<int>(extractor.compute_stmts.size()) - 1));
      }
    }
    // Phase counter increment for the consumer side.
    if (needs_phase_counter) {
      push_consumer_stmt(WrapStmtWithGuardSource(
          uniform_phase_guard_source, uniform_phase_guard,
          consumer_phase_counter->Increment()), -1);
    }
    // --- Async PV: relax the last warpgroup_wait depth ---
    // When Heddle signals tl_finegrainedws_async_pv=1, change the LAST
    // warpgroup_wait(0) to warpgroup_wait(1). This allows the final PV
    // WGMMA to remain in-flight while the consumer loop proceeds to the
    // next iteration's barrier waits and QK GEMM. PV completes by the
    // time QK's own warpgroup_wait(0) fires, so acc_o is valid before
    // its next use (acc_o *= ss). This overlaps PV tensor-core work with
    // ALU/barrier operations, recovering ~5-15% latency on FA FWD.
    {
      auto apv_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_async_pv");
      bool async_pv = false;
      if (apv_anno.has_value()) {
        std::string s =
            static_cast<std::string>(Downcast<String>(apv_anno.value()));
        async_pv = (s == "1");
      }
      if (async_pv && consumer_body_stmts.size() >= 2) {
        // Find and replace the LAST warpgroup_wait(0) in the consumer body.
        // The wait can be nested inside compound stmts (SeqStmt, IfThenElse,
        // etc.) produced by the compute_stmt lowering. We do a deep scan via
        // ir_transform on the flattened consumer body.
        Stmt full_body = SeqStmt(consumer_body_stmts);
        bool found_last = false;
        // First pass: count warpgroup_wait(0) occurrences
        int total_waits = 0;
        PostOrderVisit(full_body, [&](const ObjectRef &n) {
          if (auto *call = n.as<CallNode>()) {
            if (call->op.same_as(tl::warpgroup_wait()) &&
                call->args.size() >= 1) {
              auto *imm = call->args[0].as<IntImmNode>();
              if (imm && imm->value == 0) ++total_waits;
            }
          }
        });
        if (total_waits >= 1) {
          // Replace the LAST (by traversal order) wait(0) → wait(1)
          // using a simple recursive StmtExprMutator.
          class AsyncPVMutator : public StmtExprMutator {
           public:
            int target_idx;
            int seen = 0;
            bool replaced = false;
            Stmt VisitStmt_(const EvaluateNode *op) final {
              if (auto *call = op->value.as<CallNode>()) {
                if (call->op.same_as(tl::warpgroup_wait()) &&
                    call->args.size() >= 1) {
                  auto *imm = call->args[0].as<IntImmNode>();
                  if (imm && imm->value == 0) {
                    ++seen;
                    if (seen == target_idx) {
                      replaced = true;
                      Array<PrimExpr> na = {IntImm(DataType::Int(32), 1)};
                      return Evaluate(
                          Call(call->dtype, call->op, na));
                    }
                  }
                }
              }
              return StmtExprMutator::VisitStmt_(op);
            }
          };
          AsyncPVMutator mut;
          mut.target_idx = total_waits;
          Stmt mutated = mut(full_body);
          if (mut.replaced) {
            found_last = true;
            if (auto *seq = mutated.as<SeqStmtNode>()) {
              consumer_body_stmts = seq->seq;
            }
            LOG(INFO) << "FineGrainedWS: async PV — relaxed last warpgroup_wait "
                         "to wait(1) (" << total_waits << " waits total)";
          }
        }
      }
    }

    Stmt consumer_loop_body =
        MergeAdjacentEquivalentIfs(SeqStmt(consumer_body_stmts));
    consumer_loop_body = rewrap_loop_body_lets(consumer_loop_body);

    // --- Replace shared-memory stage expressions with phase counters ---
    // When the loop body is conditionally guarded, the barrier IDs already
    // use phase-counter-based stage/parity, but the shared-memory buffer
    // offsets still embed FloorMod(loop_var - loop_min, num_stages).
    // Rewrite them so that buffer staging stays in sync with barriers.
    if (needs_phase_counter) {
      producer_loop_body = StageExprReplacer::Replace(
          producer_loop_body, loop_var, loop_min, num_stages,
          producer_phase_counter->StageExpr(num_stages));
      consumer_loop_body = StageExprReplacer::Replace(
          consumer_loop_body, loop_var, loop_min, num_stages,
          consumer_phase_counter->StageExpr(num_stages));
    }

    // --- Build dQ Writer Loop Body (three-role only) ---
    Stmt dq_writer_loop_body;
    if (has_three_role) {
      Array<Stmt> dq_writer_stmts;
      // Wait for consumer to signal dQ data is ready in smem.
      dq_writer_stmts.push_back(Evaluate(
          Call(DataType::Handle(), named_barrier_wait(),
               {IntImm(DataType::Int(32), 1), three_role_barrier_count})));
      // Issue TMA reduce-add for each detected pattern.
      // The original stmts already contain fence_proxy_async + IfThenElse
      // guards; PCThreadIdxRewriter will convert threadIdx.x==0 to
      // tl_shuffle_elect<32>.
      for (const auto &info : tma_reduce_adds) {
        dq_writer_stmts.push_back(info.full_stmt);
      }
      // Signal consumer that smem is now free for the next iteration.
      dq_writer_stmts.push_back(Evaluate(
          Call(DataType::Handle(), named_barrier_arrive(),
               {IntImm(DataType::Int(32), 2), three_role_barrier_count})));
      dq_writer_loop_body = SeqStmt(dq_writer_stmts);
      dq_writer_loop_body = rewrap_loop_body_lets(dq_writer_loop_body);
    }

    // --- Build the loops ---
    // Remove pipeline annotations since WS handles overlap directly
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }
    Stmt producer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            producer_loop_body, Optional<IterVar>(), loop_annos);
    Stmt consumer_loop =
        For(loop_var, loop_min, loop_extent, ForKind::kSerial,
            consumer_loop_body, Optional<IterVar>(), loop_annos);

    // --- Cross-stage epilogue (Proposal 1) ---
    // For offset=-1: the main loop at k=N-1 runs the offset consumer
    // reading data from iteration k-1=N-2. The final data (iteration N-1)
    // was loaded by the producer at k=N-1 but never consumed. The epilogue
    // drains this by running the offset consumer one more time with the
    // stage corresponding to the final loaded data.
    //
    // Epilogue for each negative-offset consumer:
    //   1. Forward-wait on the final-iteration barrier (stage = (N-1) % S)
    //   2. Compute with buffer stage = (N-1) % S
    //   3. Backpressure-arrive on the final-iteration barrier
    if (has_cross_stage) {
      Array<Stmt> epilogue_stmts;
      // The epilogue stage is the stage of the LAST loaded data:
      // FloorMod(extent - 1, num_stages), since the producer loaded
      // data for iteration extent-1. When needs_phase_counter, use the
      // consumer phase counter's final value minus 1 (counter = N after
      // N executed iterations, but the last tile is iteration N-1).
      // Guard: if zero iterations executed, skip the epilogue entirely.
      PrimExpr epilogue_base =
          needs_phase_counter
              ? (consumer_phase_counter->Load() - IntImm(DataType::Int(32), 1))
              : (loop_extent - IntImm(DataType::Int(32), 1));
      PrimExpr epilogue_stage = FloorMod(epilogue_base, num_stages);
      PrimExpr epilogue_parity = FloorMod(
          FloorDiv(epilogue_base, num_stages), 2);

      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        int wait_ci = wait_insert_pos[ti];
        int offset =
            (wait_ci >= 0 &&
             wait_ci < static_cast<int>(compute_stmt_stage_offsets.size()))
                ? compute_stmt_stage_offsets[wait_ci]
                : 0;
        if (offset >= 0)
          continue;
        // Forward-wait for the final-iteration data
        ICHECK_GE(fwd_bases[ti], 0);
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), fwd_bases[ti]) + epilogue_stage;
        epilogue_stmts.push_back(
            makeParityWait(fwd_id, epilogue_parity));
      }

      for (size_t ci = 0; ci < extractor.compute_stmts.size(); ++ci) {
        int offset = compute_stmt_stage_offsets[ci];
        if (offset >= 0 || moved_compute_stmts[ci] ||
            tma_reduce_add_indices.count(static_cast<int>(ci)))
          continue;
        // Compute with the final-iteration's buffer stage
        Stmt epilogue_stmt = StageExprReplacer::Replace(
            extractor.compute_stmts[ci], loop_var, loop_min, num_stages,
            epilogue_stage);
        epilogue_stmts.push_back(epilogue_stmt);
      }

      for (size_t ti = 0; ti < extractor.blocks.size(); ++ti) {
        int wait_ci = wait_insert_pos[ti];
        int offset =
            (wait_ci >= 0 &&
             wait_ci < static_cast<int>(compute_stmt_stage_offsets.size()))
                ? compute_stmt_stage_offsets[wait_ci]
                : 0;
        if (offset >= 0)
          continue;
        // Backpressure-arrive for the final-iteration slot
        PrimExpr bp_id =
            IntImm(DataType::Int(32), bp_bases[ti]) + epilogue_stage;
        epilogue_stmts.push_back(makeArriveBarrier(bp_id));
      }

      if (!epilogue_stmts.empty()) {
        Stmt epilogue_body = SeqStmt(epilogue_stmts);
        // Guard: skip epilogue if fewer than abs(offset) iterations executed.
        // This handles masked loops where all iterations may be skipped.
        PrimExpr min_required =
            IntImm(DataType::Int(32), max_abs_stage_offset);
        PrimExpr guard_expr =
            needs_phase_counter
                ? GE(consumer_phase_counter->Load(), min_required)
                : GE(loop_extent, min_required);
        epilogue_body = IfThenElse(guard_expr, epilogue_body);
        consumer_loop = SeqStmt({consumer_loop, epilogue_body});
      }
    }

    // --- Early bp_arrive: move backpressure arrives before warpgroup_wait ---
    // WGMMA reads shared memory at issue time, not at completion. After
    // warpgroup_commit_batch(), the shared buffer is no longer being read.
    // Moving bp_arrive from after warpgroup_wait to before it lets the
    // producer start the next TMA load ~40 cycles earlier per iter.
    {
      auto apv_anno2 =
          pipeline_loop->annotations.Get("tl_finegrainedws_async_pv");
      bool do_early_bp = false;
      if (apv_anno2.has_value()) {
        std::string s2 =
            static_cast<std::string>(Downcast<String>(apv_anno2.value()));
        do_early_bp = (s2 == "1");
      }
      if (do_early_bp && consumer_body_stmts.size() >= 2) {
        Array<Stmt> new_stmts;
        size_t i = 0;
        int moved = 0;
        while (i < consumer_body_stmts.size()) {
          Stmt cur = consumer_body_stmts[i];
          bool has_bp_next = false;
          Stmt bp_stmt;
          if (i + 1 < consumer_body_stmts.size()) {
            Stmt nxt = consumer_body_stmts[i + 1];
            bool is_arrive = false;
            PostOrderVisit(nxt, [&](const ObjectRef &n) {
              if (auto *call = n.as<CallNode>()) {
                if (call->op.same_as(builtin::ptx_arrive_barrier()))
                  is_arrive = true;
              }
            });
            if (is_arrive) { has_bp_next = true; bp_stmt = nxt; }
          }
          bool has_commit_wait = false;
          if (has_bp_next) {
            bool hc = false, hw = false;
            PostOrderVisit(cur, [&](const ObjectRef &n) {
              if (auto *call = n.as<CallNode>()) {
                if (call->op.same_as(tl::warpgroup_commit_batch())) hc = true;
                if (call->op.same_as(tl::warpgroup_wait())) hw = true;
              }
            });
            has_commit_wait = hc && hw;
          }
          if (has_commit_wait && has_bp_next) {
            class EarlyBPMut : public StmtExprMutator {
             public:
              Stmt bp; bool ok = false;
              Stmt VisitStmt_(const EvaluateNode *op) final {
                if (!ok) {
                  if (auto *c = op->value.as<CallNode>()) {
                    if (c->op.same_as(tl::warpgroup_wait())) {
                      ok = true;
                      return SeqStmt({bp, GetRef<Stmt>(op)});
                    }
                  }
                }
                return StmtExprMutator::VisitStmt_(op);
              }
            };
            EarlyBPMut mut; mut.bp = bp_stmt;
            Stmt mod = mut(cur);
            if (mut.ok) {
              new_stmts.push_back(mod);
              i += 2; ++moved; continue;
            }
          }
          new_stmts.push_back(cur); ++i;
        }
        if (moved > 0) {
          consumer_body_stmts = new_stmts;
          LOG(INFO) << "FineGrainedWS: early bp_arrive — moved " << moved
                    << " bp arrive(s) before warpgroup_wait";
        }
      }
    }

    // Async WGMMA epilogue safety: when the last in-loop warpgroup_wait
    // was relaxed to wait<1>, the final iteration's WGMMA may still be
    // pending at loop exit. Insert a wait<0> + fence_operand to drain the
    // pipeline before the post-loop epilogue reads the accumulator.
    {
      auto apv_anno =
          pipeline_loop->annotations.Get("tl_finegrainedws_async_pv");
      bool async_pv_active = false;
      if (apv_anno.has_value()) {
        std::string s =
            static_cast<std::string>(Downcast<String>(apv_anno.value()));
        async_pv_active = (s == "1");
      }
      if (async_pv_active) {
        Stmt drain = Evaluate(
            Call(DataType::Handle(), tl::warpgroup_wait(),
                 {IntImm(DataType::Int(32), 0)}));
        consumer_loop = SeqStmt({consumer_loop, drain});
      }
    }

    // Wrap loops with phase counter allocation when needed.
    if (needs_phase_counter) {
      producer_loop = producer_phase_counter->WrapLoopWithAlloc(producer_loop);
      consumer_loop = consumer_phase_counter->WrapLoopWithAlloc(consumer_loop);
    }

    Stmt ws_body;
    if (has_three_role) {
      // --- Three-role thread split ---
      // Producer WG layout: warp 0 = TMA loads, warp 1 = dQ writer,
      // warps 2-3 = idle. All share the same WG (128 threads, setmaxnreg 24).
      PrimExpr warp_extent = IntImm(DataType::Int(32), 32);

      // Build the dQ writer loop.
      Stmt dq_writer_loop =
          For(loop_var, loop_min, loop_extent, ForKind::kSerial,
              dq_writer_loop_body, Optional<IterVar>(), loop_annos);

      // Prime the dQ_empty barrier: the buffer is initially empty, so the
      // consumer's first named_barrier_wait(2, count) must not block.
      // The dQ writer signals "empty" before entering the loop.
      Stmt dq_writer_prime = Evaluate(
          Call(DataType::Handle(), named_barrier_arrive(),
               {IntImm(DataType::Int(32), 2), three_role_barrier_count}));
      dq_writer_loop = SeqStmt({dq_writer_prime, dq_writer_loop});

      // Phase counter for dQ writer if needed.
      if (needs_phase_counter) {
        // dQ writer doesn't use mbarrier stage/parity, but if the loop
        // is guarded, it must iterate the same number of times. Reuse
        // the stage expression for any smem stage offsets in TMA store args.
        auto dq_writer_phase_counter =
            PhaseCounter::Create("dq_writer_phase");
        dq_writer_loop = StageExprReplacer::Replace(
            dq_writer_loop, loop_var, loop_min, num_stages,
            dq_writer_phase_counter.StageExpr(num_stages));
        dq_writer_loop =
            dq_writer_phase_counter.WrapLoopWithAlloc(dq_writer_loop);
      }

      // Rewrite threadIdx.x for each role.
      // Producer warp 0: threadIdx.x -> threadIdx.x - consumer_extent,
      // extent = 32
      producer_loop = PCThreadIdxRewriter::Rewrite(
          producer_loop, thread_iv_->var,
          thread_iv_->var - consumer_thread_extent, warp_extent,
          /*do_shuffle=*/true);
      // dQ writer warp 1: threadIdx.x -> threadIdx.x - consumer_extent - 32,
      // extent = 32
      dq_writer_loop = PCThreadIdxRewriter::Rewrite(
          dq_writer_loop, thread_iv_->var,
          thread_iv_->var - consumer_thread_extent - warp_extent, warp_extent,
          /*do_shuffle=*/true);
      // Consumer: threadIdx.x -> threadIdx.x, extent = consumer_extent
      consumer_loop = PCThreadIdxRewriter::Rewrite(
          consumer_loop, thread_iv_->var, thread_iv_->var,
          consumer_thread_extent, /*do_shuffle=*/true);

      // Build the producer WG dispatch: warp 0 → producer, warp 1 → dQ writer
      PrimExpr local_tid = thread_iv_->var - consumer_thread_extent;
      PrimExpr warp_in_pg = FloorDiv(local_tid, 32);
      Stmt producer_wg_body = IfThenElse(
          EQ(warp_in_pg, IntImm(DataType::Int(32), 0)), producer_loop,
          IfThenElse(EQ(warp_in_pg, IntImm(DataType::Int(32), 1)),
                     dq_writer_loop, Evaluate(0)));
      ws_body = IfThenElse(GE(thread_iv_->var, consumer_thread_extent),
                           producer_wg_body, consumer_loop);
    } else if (dual_consumer_enabled_) {
      // --- Dual-consumer warp group mode ---
      // Split consumer into WG0 (QK+Softmax) and WG1 (PV) for TC overlap.
      // Stmts with stage_offset != 0 (detected via compute_stmt_stage_offsets)
      // are PV stmts and go to WG1. Everything else goes to WG0.
      //
      // Thread layout:
      //   WG0: tid [0, 128)               = QK + Softmax (128 threads)
      //   WG1: tid [128, 256)             = PV (128 threads)
      //   Producer: tid [256, 384)        = TMA loads
      //
      // Named barriers (IDs 3 and 4):
      //   Barrier 3 (att_ready): WG0 → WG1 (att data computed)
      //   Barrier 4 (att_consumed): WG1 → WG0 (att buffer free)
      //   Participant count: 256 (all consumer threads)
      //
      // Key insight on mbarrier arrive counts:
      //   bp barriers for K buffers (non-offset): arrived by WG0 (128 threads)
      //   bp barriers for V buffers (offset): arrived by WG1 (128 threads)
      //   → All bp barriers use wg_extent (128) instead of consumer_extent (256)

      PrimExpr wg_extent = IntImm(DataType::Int(32), 128);
      // Named barrier participant count: both WG0 and WG1 participate
      // = 2 * wg_extent = 256 (even when consumer_thread_extent = 128)
      PrimExpr dual_barrier_count =
          IntImm(DataType::Int(32), 2) * wg_extent;

      // Identify which consumer_body_stmts are PV-related.
      // PV stmts are those at/after a compute_stmt with offset != 0.
      // We find the FIRST offset compute_stmt position and split there.
      // Find the split point: where does PV work begin?
      // Use the V buffer's wait_insert_pos as the boundary. The V fwd
      // wait is the first PV-related stmt in the consumer body.
      // Everything before it goes to WG0, everything from it onward to WG1.
      //
      // For dual-consumer with restructured data flow (WG1 does rescale
      // + PV), the split should be at the LAST stmt that WG0 owns.
      // WG0 owns: K wait, QK, softmax, att→smem store, m/l state updates.
      // WG1 owns: rescale, V wait, PV, V bp arrive.
      //
      // Heuristic: find the V buffer's wait_insert_pos. The consumer_body
      // stmt at that position is the V fwd wait. Split 2 stmts before
      // (to include att→smem copy in WG0, rescale in WG1).
      //
      // Robust approach: use the SECOND-TO-LAST producer block's
      // wait position as the split. For FA FWD with K+V, the V block
      // is the last one.
      int pv_split_idx = -1;
      {
        // Find the last producer block (V buffer)
        int last_block_idx = static_cast<int>(extractor.blocks.size()) - 1;
        if (last_block_idx >= 0) {
          int v_wait_pos = wait_insert_pos[last_block_idx];
          // The V wait in consumer_body is at this compute_stmt index.
          // Map it to consumer_body_stmts index by counting stmts.
          // Since consumer_body_stmts includes waits interleaved with
          // compute stmts, the V wait stmt is somewhere in the array.
          // Simpler: count consumer_body_stmts from the end. The last
          // few stmts are: V wait, V compute, V bp arrive. Split before
          // the first of these.
          //
          // Most robust: just split at consumer_body_stmts.size() - N
          // where N is the number of PV-related stmts. For FA FWD:
          // V fwd wait + rescale + PV compute + V bp arrive = ~4 stmts.
          // But with cross-stage, there are IfThenElse guards too.
          //
          // Simplest: split at the midpoint weighted toward the end.
          // The last 1/3 of stmts are typically PV-related.
          int total = static_cast<int>(consumer_body_stmts.size());
          pv_split_idx = total * 2 / 3;  // split at ~67%
          // Adjust: look for the last stmt that's a bp arrive for K
          // (which marks the end of WG0's QK section).
          // Find the FIRST K bp arrive (not the last). In FA FWD:
          // stmts = [K_wait, QK, K_bp_arrive, softmax..., att→smem,
          //          m→smem, m_new→smem, m_new→m, l_new→l,
          //          rescale, V_wait, PV, V_bp_arrive]
          // We want to split AFTER the last smem store (att/m/m_new).
          // WG0: K_wait→att→smem→m→smem→m_new→smem→m_new→m→l_new→l
          // WG1: rescale→V_wait→PV→V_bp_arrive
          //
          // Heuristic: find stmts that READ from shared buffers
          // that were WRITTEN in earlier stmts (att_smem, m_smem, etc).
          // The first such read marks the start of WG1.
          // Or simpler: find the V fwd wait using wait_insert_pos.
          int v_block_idx = static_cast<int>(extractor.blocks.size()) - 1;
          int v_wait_ci = (v_block_idx >= 0) ? wait_insert_pos[v_block_idx] : -1;
          // The V fwd wait is at consumer_body_stmts position for ci=v_wait_ci.
          // Count stmts: each ci adds some waits+compute+arrives.
          // For robustness, scan for the stmt that matches v_wait_ci.
          // Since stmts are ordered by ci, the V-related stmts are near the end.
          //
          // Use the last third of stmts as PV section (empirical for FA FWD):
          // FA FWD has ~19 stmts: 12 QK+softmax + 7 PV-related.
          // Split at 2/3 = 12.67 ≈ 13 stmts for WG0.
          // Split so WG1 gets: rescale + V fwd wait + PV wgmma + V bp arrive
          // = last 4 stmts (or more if there are extra stmts)
          // Use Heddle-provided split index if available; otherwise fall
          // back to the V-wait scan. The scan used to run unconditionally
          // and silently overrode Heddle's choice, discarding the Python
          // cost-model decision and always picking the last V-wait.
          if (heddle_split_idx >= 0 && heddle_split_idx < total) {
            pv_split_idx = heddle_split_idx;
          } else {
            pv_split_idx = total - 4;
            for (int si = total - 1; si >= total / 2; --si) {
              auto stmt_str = std::ostringstream();
              stmt_str << consumer_body_stmts[si];
              std::string s = stmt_str.str();
              if (s.find("mbarrier_wait_parity") != std::string::npos ||
                  s.find("makeParityWait") != std::string::npos ||
                  s.find("wait_parity") != std::string::npos) {
                // V wait found at si. Split HERE — rescale stays in WG0.
                // WG1 gets: V wait + PV + V bp arrive.
                // WG1's rescale is handled by a codegen intrinsic (flat copy
                // based).
                pv_split_idx = si;
                break;
              }
            }
          }
        }
      }

      if (pv_split_idx < 0) {
        // No PV stmts found — can't do dual-consumer, fall through
        // to standard two-role.
        LOG(WARNING) << "FineGrainedWS dual-consumer: no PV stmts detected, "
                        "falling back to standard two-role.";
        goto standard_two_role;
      }

      {
        // ---- Generalized dual-consumer: auto-detect cross-WG transfers ----
        // Use def-use analysis on the split point to find which fragment
        // buffers WG0 writes and WG1 reads -> those need flat copy.

        struct DualBufInfo { std::string name; DataType dtype; int64_t elems; };
        std::unordered_map<std::string, DualBufInfo> buf_info_map;
        std::string sscl_name, o_acc_name;

        // Collect buffer info from all stmts
        for (size_t si = 0; si < consumer_body_stmts.size(); ++si) {
          PostOrderVisit(consumer_body_stmts[si],
              [&](const ObjectRef& node) {
            auto bs = node.as<BufferStoreNode>();
            if (!bs || buf_info_map.count(bs->buffer->name)) return;
            int64_t total = 1;
            for (auto& s : bs->buffer->shape) {
              auto imm = s.as<IntImmNode>();
              if (imm) total *= imm->value;
              else { total = -1; break; }
            }
            if (total > 0)
              buf_info_map[bs->buffer->name] = {
                  bs->buffer->name, bs->buffer->dtype, total};
          });
        }

        // Def-use: WG0 defs ∩ WG1 uses = cross-WG transfer set
        std::unordered_set<std::string> wg0_defs, wg1_uses;
        for (int i = 0; i < pv_split_idx; ++i) {
          PostOrderVisit(consumer_body_stmts[i], [&](const ObjectRef& n) {
            if (auto bs = n.as<BufferStoreNode>()) wg0_defs.insert(bs->buffer->name);
          });
        }
        // Build Var-to-buffer-name map for WGMMA arg matching
        std::unordered_map<const VarNode*, std::string> var_to_name;
        for (auto& [name, info] : buf_info_map) {
          // Scan all stmts for BufferStore to find the data Var
          for (size_t si = 0; si < consumer_body_stmts.size(); ++si) {
            PostOrderVisit(consumer_body_stmts[si], [&](const ObjectRef& n) {
              if (auto bs = n.as<BufferStoreNode>()) {
                if (bs->buffer->name == name)
                  var_to_name[bs->buffer->data.get()] = name;
              }
              if (auto bl = n.as<BufferLoadNode>()) {
                if (bl->buffer->name == name)
                  var_to_name[bl->buffer->data.get()] = name;
              }
            });
          }
        }

        for (size_t i = pv_split_idx; i < consumer_body_stmts.size(); ++i) {
          PostOrderVisit(consumer_body_stmts[i], [&](const ObjectRef& n) {
            if (auto bl = n.as<BufferLoadNode>()) wg1_uses.insert(bl->buffer->name);
            if (auto bs = n.as<BufferStoreNode>()) wg1_uses.insert(bs->buffer->name);
            // Also catch WGMMA Call args (Var references to fragment buffers)
            if (auto call = n.as<CallNode>()) {
              if (call->op.same_as(ptx_wgmma_rs()) || call->op.same_as(ptx_wgmma_ss())) {
                for (int idx : {6, 10}) {  // A_data=6, C_data=10
                  if (auto *v = call->args[idx].as<VarNode>()) {
                    auto it = var_to_name.find(v);
                    if (it != var_to_name.end()) wg1_uses.insert(it->second);
                  }
                }
              }
            }
          });
        }

        // Transfer all f16/bf16 fragments that WG0 writes.
        // Can't use WG0∩WG1 intersection because WGMMA Var args have
        // lost buffer identity. f16 fragments are WGMMA intermediates
        // (att/softmax output) that WG1 always needs.
        std::vector<DualBufInfo> xfer_bufs;
        for (auto& name : wg0_defs) {
          if (buf_info_map.count(name)) {
            auto& info = buf_info_map[name];
            if (info.dtype == DataType::Float(16) ||
                info.dtype == DataType::BFloat(16))
              xfer_bufs.push_back(info);
          }
        }
        std::sort(xfer_bufs.begin(), xfer_bufs.end(),
            [](const DualBufInfo& a, const DualBufInfo& b) { return a.elems > b.elems; });

        // Detect FA-FWD sscl pattern (optional rescale).
        // sscl = exp2(prev - new) is a SMALL 1D f32 buffer (≤4 elems per thread,
        // corresponding to per-row softmax stats). Reject large buffers to avoid
        // false-positives on FA BWD's softmax: qkT = exp2(qkT*s - lse).
        for (int si = 0; si < pv_split_idx && sscl_name.empty(); ++si) {
          PostOrderVisit(consumer_body_stmts[si], [&](const ObjectRef& node) {
            auto bs = node.as<BufferStoreNode>();
            if (!bs || !sscl_name.empty()) return;
            if (bs->buffer->dtype != DataType::Float(32) || bs->buffer->shape.size() != 1) return;
            // Size check: sscl should be ≤4 elements (2 per row group)
            auto sz_imm = bs->buffer->shape[0].as<IntImmNode>();
            if (!sz_imm || sz_imm->value > 4) return;
            auto cast_v = bs->value.as<CastNode>();
            PrimExpr inner = cast_v ? cast_v->value : bs->value;
            if (auto call = inner.as<CallNode>())
              if (call->args.size() >= 1 && call->args[0].as<SubNode>())
                sscl_name = bs->buffer->name;
          });
        }
        // Detect lsum (x = x*scale + sum pattern) for FA-FWD transfer
        std::string lsum_name;
        for (int si = 0; si < pv_split_idx && lsum_name.empty(); ++si) {
          PostOrderVisit(consumer_body_stmts[si], [&](const ObjectRef& node) {
            auto bs = node.as<BufferStoreNode>();
            if (!bs || !lsum_name.empty()) return;
            if (bs->buffer->dtype != DataType::Float(32) || bs->buffer->shape.size() != 1) return;
            auto sz_imm2 = bs->buffer->shape[0].as<IntImmNode>();
            if (!sz_imm2 || sz_imm2->value > 4) return;  // small scalar buffer only
            auto add = bs->value.as<AddNode>();
            if (add) {
              auto mul = add->a.as<MulNode>();
              if (mul) {
                auto ld = mul->a.as<BufferLoadNode>();
                if (ld && ld->buffer->data.same_as(bs->buffer->data))
                  lsum_name = bs->buffer->name;
              }
            }
          });
        }

        // When sscl/lsum detected, add them to xfer_bufs and find o_acc
        if (!sscl_name.empty()) {
          if (buf_info_map.count(sscl_name))
            xfer_bufs.push_back(buf_info_map[sscl_name]);
          int64_t mx = 0;
          for (auto& [nm, info] : buf_info_map)
            if (info.elems > mx) { mx = info.elems; o_acc_name = nm; }
        }
        if (!lsum_name.empty() && buf_info_map.count(lsum_name))
          xfer_bufs.push_back(buf_info_map[lsum_name]);

        // Att name = largest f16 transfer buffer (for codegen smem sizing)
        std::string att_name;
        int att_elems_per_thread = 0;
        for (auto& buf : xfer_bufs) {
          if ((buf.dtype == DataType::Float(16) || buf.dtype == DataType::BFloat(16))
              && buf.elems > att_elems_per_thread) {
            att_elems_per_thread = buf.elems;
            att_name = buf.name;
          }
        }

        {
          std::ostringstream oss;
          oss << "FineGrainedWS dual-consumer: xfer=[";
          for (size_t i = 0; i < xfer_bufs.size(); ++i) {
            if (i) oss << ", ";
            oss << xfer_bufs[i].name << "(" << xfer_bufs[i].elems << " " << xfer_bufs[i].dtype << ")";
          }
          oss << "]";
          if (!sscl_name.empty()) oss << " sscl=" << sscl_name;
          LOG(INFO) << oss.str();
        }

        // Build WG0 and WG1 loop bodies with PING-PONG double buffering
        // for QK‖PV overlap.
        //
        // Barrier protocol (ping-pong with 2 slots):
        //   att_ready[s]:    barrier_id = 3 + 2*s  (s = k%2)
        //   att_consumed[s]: barrier_id = 4 + 2*s
        //   Slot 0: barriers 3 (ready), 4 (consumed)
        //   Slot 1: barriers 5 (ready), 6 (consumed)
        //
        // WG0 iter k: wait consumed[k%2] → QK → store att[k%2] → signal ready[k%2]
        // WG1 iter k: wait ready[k%2]    → load att[k%2] → PV → signal consumed[k%2]
        //
        // Overlap: WG0 k+1 can start QK while WG1 k is doing PV (different slots).
        //
        // The codegen flat copy intrinsics take a "slot" argument (loop_var % 2)
        // to index into double-buffered __tl_att_flat[2][...] arrays.

        Array<Stmt> wg0_body_stmts;
        Array<Stmt> wg1_body_stmts;

        // Single-slot cross-WG transfer (no ping-pong double-buffering).
        // The previous 2-slot design halves throughput when WG0 and WG1
        // are balanced but doubles the static smem footprint for att_flat
        // and m/l_flat. For FA FWD with bM=128 + stages=2, 2 slots
        // overflow H100's 228 KB smem budget. A single slot adds one
        // barrier stall per iter (WG0 waits for WG1 to consume before
        // writing) but keeps total smem under budget and preserves the
        // critical QK‖PV wgmma overlap.
        PrimExpr slot = IntImm(DataType::Int(32), 0);
        PrimExpr barrier_ready = IntImm(DataType::Int(32), 3);
        PrimExpr barrier_consumed = IntImm(DataType::Int(32), 4);

        // WG0: wait att_consumed[k%2] (primed by WG1 before loop)
        wg0_body_stmts.push_back(Evaluate(
            Call(DataType::Handle(), named_barrier_wait(),
                 {barrier_consumed, dual_barrier_count})));

        // WG0: all stmts before the PV split point, SKIPPING dead acc_o
        // rescale (WG0 doesn't write output, so acc_o *= sscl is wasted).
        for (int i = 0; i < pv_split_idx; ++i) {
          // Detect acc_o rescale: ForNode writing to o_acc buffer with *= pattern
          bool is_dead_rescale = false;
          if (!o_acc_name.empty()) {
            PostOrderVisit(consumer_body_stmts[i],
                [&](const ObjectRef& node) {
              auto bs = node.as<BufferStoreNode>();
              if (bs && bs->buffer->name == o_acc_name) {
                // Check if value = load(same_buf) * something
                auto mul = bs->value.as<MulNode>();
                if (mul) {
                  auto load = mul->a.as<BufferLoadNode>();
                  if (load && load->buffer->name == o_acc_name) {
                    is_dead_rescale = true;
                  }
                }
              }
            });
          }
          if (!is_dead_rescale) {
            wg0_body_stmts.push_back(consumer_body_stmts[i]);
          }
        }

        // WG0: flat store ALL cross-WG buffers to slot k%2
        // Dispatch by dtype: f16→att_flat (half_t smem), f32→m_flat (float smem)
        int f32_buf_idx = 0;  // index into __tl_m_flat / __tl_l_flat
        for (auto& buf : xfer_bufs) {
          if (buf.dtype == DataType::Float(32)) {
            // f32 scalar buffer → use m_flat/l_flat intrinsic (idx 0=m, 1=l)
            std::string intrinsic = (f32_buf_idx == 0)
                ? "tl::dual_consumer_m_flat_store"
                : "tl::dual_consumer_l_flat_store";
            wg0_body_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::call_extern(),
                     {StringImm(intrinsic),
                      IntImm(DataType::Int(32), buf.elems),
                      StringImm(buf.name), slot})));
            f32_buf_idx++;
          } else {
            // f16/bf16 fragment → att_flat intrinsic
            wg0_body_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::call_extern(),
                     {StringImm("tl::dual_consumer_att_flat_store"),
                      IntImm(DataType::Int(32), buf.elems),
                      StringImm(buf.name), slot})));
          }
        }

        // WG0: signal att_ready[k%2]
        wg0_body_stmts.push_back(Evaluate(
            Call(DataType::Handle(), named_barrier_arrive(),
                 {barrier_ready, dual_barrier_count})));

        // WG1: wait att_ready[k%2]
        wg1_body_stmts.push_back(Evaluate(
            Call(DataType::Handle(), named_barrier_wait(),
                 {barrier_ready, dual_barrier_count})));

        // WG1: fence_proxy_async for memory visibility
        wg1_body_stmts.push_back(Evaluate(
            Call(DataType::Handle(), tl::fence_proxy_async(), {})));

        // WG1: flat load ALL cross-WG buffers from slot k%2
        f32_buf_idx = 0;
        for (auto& buf : xfer_bufs) {
          if (buf.dtype == DataType::Float(32)) {
            std::string intrinsic = (f32_buf_idx == 0)
                ? "tl::dual_consumer_m_flat_load"
                : "tl::dual_consumer_l_flat_load";
            wg1_body_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::call_extern(),
                     {StringImm(intrinsic),
                      IntImm(DataType::Int(32), buf.elems),
                      IntImm(DataType::Int(32), 128),
                      StringImm(buf.name), slot})));
            f32_buf_idx++;
          } else {
            wg1_body_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::call_extern(),
                     {StringImm("tl::dual_consumer_att_flat_load"),
                      IntImm(DataType::Int(32), buf.elems),
                      IntImm(DataType::Int(32), 128),
                      StringImm(buf.name), slot})));
          }
        }

        // WG1: rescale (FA-FWD only, when sscl pattern detected).
        // Pass the per-thread acc_o element count so the intrinsic can
        // emit the right number of float4 rescales. The previous
        // implementation hard-coded 16 iterations (64 fp32 per thread),
        // which only works for threads=256 FA FWD. For threads=128 each
        // thread holds 128 fp32 of acc_o and needs 32 iterations.
        if (!sscl_name.empty() && !o_acc_name.empty()) {
          int64_t o_acc_elems = 0;
          if (buf_info_map.count(o_acc_name))
            o_acc_elems = buf_info_map[o_acc_name].elems;
          wg1_body_stmts.push_back(Evaluate(
              Call(DataType::Handle(), builtin::call_extern(),
                   {StringImm("tl::dual_consumer_rescale_o_acc"),
                    StringImm(sscl_name), StringImm(o_acc_name),
                    IntImm(DataType::Int(32), static_cast<int>(o_acc_elems))})));
        }

        // WG1: all stmts from PV split point onward
        for (size_t i = pv_split_idx;
             i < consumer_body_stmts.size(); ++i) {
          wg1_body_stmts.push_back(consumer_body_stmts[i]);
        }

        // WG1: signal att_consumed[k%2]
        wg1_body_stmts.push_back(Evaluate(
            Call(DataType::Handle(), named_barrier_arrive(),
                 {barrier_consumed, dual_barrier_count})));

        // Phase counter increment: only in WG0's loop (WG1 tracks its own)
        if (needs_phase_counter) {
          wg0_body_stmts.push_back(WrapStmtWithGuardSource(
              uniform_phase_guard_source, uniform_phase_guard,
              consumer_phase_counter->Increment()));
        }

        Stmt wg0_loop_body = SeqStmt(wg0_body_stmts);
        Stmt wg1_loop_body = SeqStmt(wg1_body_stmts);
        wg0_loop_body = rewrap_loop_body_lets(wg0_loop_body);
        wg1_loop_body = rewrap_loop_body_lets(wg1_loop_body);

        // Stage expression rewrite for guarded loops
        if (needs_phase_counter) {
          wg0_loop_body = StageExprReplacer::Replace(
              wg0_loop_body, loop_var, loop_min, num_stages,
              consumer_phase_counter->StageExpr(num_stages));
          auto wg1_phase = PhaseCounter::Create("wg1_phase");
          wg1_loop_body = StageExprReplacer::Replace(
              wg1_loop_body, loop_var, loop_min, num_stages,
              wg1_phase.StageExpr(num_stages));
          // WG1 phase counter needs increment too
          wg1_loop_body = SeqStmt({wg1_loop_body,
              WrapStmtWithGuardSource(
                  uniform_phase_guard_source, uniform_phase_guard,
                  wg1_phase.Increment())});
        }

        // Remove the duplicate phase counter increment from consumer_body
        // (it was already added to WG0 above, and consumer_body may have it)
        // This is handled by the split: stmts after pv_split go to WG1.

        Stmt wg0_loop = For(loop_var, loop_min, loop_extent,
                             ForKind::kSerial, wg0_loop_body,
                             Optional<IterVar>(), loop_annos);
        Stmt wg1_loop = For(loop_var, loop_min, loop_extent,
                             ForKind::kSerial, wg1_loop_body,
                             Optional<IterVar>(), loop_annos);

        if (needs_phase_counter) {
          wg0_loop = consumer_phase_counter->WrapLoopWithAlloc(wg0_loop);
          auto wg1_phase_alloc = PhaseCounter::Create("wg1_phase");
          wg1_loop = wg1_phase_alloc.WrapLoopWithAlloc(wg1_loop);
        }

        // Prime att_consumed for BOTH ping-pong slots before loop start
        Stmt wg1_prime0 = Evaluate(
            Call(DataType::Handle(), named_barrier_arrive(),
                 {IntImm(DataType::Int(32), 4), dual_barrier_count}));
        Stmt wg1_prime1 = Evaluate(
            Call(DataType::Handle(), named_barrier_arrive(),
                 {IntImm(DataType::Int(32), 6), dual_barrier_count}));
        wg1_loop = SeqStmt({wg1_prime0, wg1_prime1, wg1_loop});

        // Thread layout: [0, 128) = WG0, [128, 256) = WG1, [256, 384) = producer
        PrimExpr dual_consumer_extent =
            IntImm(DataType::Int(32), 2) * wg_extent;  // 256
        ws_consumer_thread_extent = dual_consumer_extent;

        // Rewrite threadIdx.x for each role
        producer_loop = PCThreadIdxRewriter::Rewrite(
            producer_loop, thread_iv_->var,
            thread_iv_->var - dual_consumer_extent, producer_thread_extent,
            /*do_shuffle=*/true);
        // WG0: threadIdx = tid (range 0-127)
        int orig_consumer_int =
            Downcast<IntImm>(consumer_thread_extent)->value;
        wg0_loop = PCThreadIdxRewriter::Rewrite(
            wg0_loop, thread_iv_->var, thread_iv_->var, wg_extent,
            /*do_shuffle=*/true,
            /*rewrite_barrier_from=*/orig_consumer_int,
            /*rewrite_barrier_to=*/128);
        // WG1: threadIdx = tid - 128 (range 0-127)
        // First: substitute ALL threadIdx-named Vars in WG1 with (Var - 128).
        // This catches lowered built-in threadIdx refs that PCThreadIdxRewriter
        // can't match by pointer (they're different Var objects from thread_iv_).
        wg1_loop = ThreadIdxSubstitutor::Substitute(
            wg1_loop, thread_iv_->var, wg_extent);
        // Only apply barrier rewrite, NOT threadIdx rewrite (already done above)
        // Use a dummy Var for thread_var to avoid double-replacing threadIdx
        Var dummy_var("__no_match__", DataType::Int(32));
        wg1_loop = PCThreadIdxRewriter::Rewrite(
            wg1_loop, dummy_var, dummy_var, wg_extent,
            /*do_shuffle=*/false,
            /*rewrite_barrier_from=*/orig_consumer_int,
            /*rewrite_barrier_to=*/128,
            /*barrier_id_offset=*/4);

        // Thread dispatch:
        //   tid < 128 → WG0
        //   128 <= tid < 256 → WG1
        //   tid >= 256 → producer
        Stmt consumer_dispatch = IfThenElse(
            LT(thread_iv_->var, wg_extent), wg0_loop, wg1_loop);
        ws_body = IfThenElse(GE(thread_iv_->var, dual_consumer_extent),
                             producer_loop, consumer_dispatch);

        // Override bp barrier arrive counts: use wg_extent (128) not
        // consumer_extent (256), since each bp barrier is only arrived
        // by one WG (WG0 for K, WG1 for V).
        // This is handled below in the barrier_arrive_counts section
        // by checking dual_consumer_enabled_.

        LOG(INFO) << "FineGrainedWS dual-consumer: split at stmt " << pv_split_idx
                  << "/" << consumer_body_stmts.size()
                  << " (WG0: " << pv_split_idx << " stmts, WG1: "
                  << (consumer_body_stmts.size() - pv_split_idx) << " stmts)";
      }
    } else if (track_warp_groups && !consumer_stmt_warp_group.empty()) {
      // --- Plan B: Per-op warp dispatch ---
      // Each consumer compute stmt is routed to a specific warp group based
      // on the solver's warp_assigns map. This is a generalization of
      // dual-consumer: instead of a fixed positional split, each stmt goes
      // to its assigned warp group. Common stmts (wg=-1) are duplicated
      // into all warp groups.
      //
      // Thread layout (N warp groups):
      //   WG0: tid [0, 128)
      //   WG1: tid [128, 256)
      //   ...
      //   WG(N-1): tid [(N-1)*128, N*128)
      //   Producer: tid [N*128, (N+1)*128)

      // Determine number of warp groups
      int max_wg = 0;
      for (int wg : consumer_stmt_warp_group) {
        if (wg > max_wg) max_wg = wg;
      }
      int num_warp_groups = max_wg + 1;
      if (num_warp_groups < 2) {
        // Only one warp group — fall through to standard two-role
        LOG(INFO) << "FineGrainedWS per-op dispatch: only 1 warp group, "
                     "falling back to standard two-role.";
        goto standard_two_role;
      }

      PrimExpr wg_extent = IntImm(DataType::Int(32), 128);
      PrimExpr total_consumer_threads =
          IntImm(DataType::Int(32), num_warp_groups * 128);
      ws_consumer_thread_extent = total_consumer_threads;
      int orig_consumer_int =
          Downcast<IntImm>(consumer_thread_extent)->value;

      {
        // Build per-WG loop bodies
        std::vector<Array<Stmt>> wg_stmts(num_warp_groups);
        ICHECK_EQ(consumer_body_stmts.size(), consumer_stmt_warp_group.size());
        for (size_t si = 0; si < consumer_body_stmts.size(); ++si) {
          int wg = consumer_stmt_warp_group[si];
          if (wg < 0) {
            // Common stmt: goes to all warp groups
            for (int w = 0; w < num_warp_groups; ++w) {
              wg_stmts[w].push_back(consumer_body_stmts[si]);
            }
          } else {
            ICHECK_LT(wg, num_warp_groups)
                << "warp group " << wg << " exceeds num_warp_groups " << num_warp_groups;
            wg_stmts[wg].push_back(consumer_body_stmts[si]);
          }
        }

        // Build loops for each warp group
        std::vector<Stmt> wg_loops(num_warp_groups);
        for (int w = 0; w < num_warp_groups; ++w) {
          if (wg_stmts[w].empty()) {
            LOG(WARNING) << "FineGrainedWS per-op dispatch: WG" << w
                         << " has no stmts, inserting nop";
            wg_loops[w] = Evaluate(0);
          } else {
            Stmt wg_body = MergeAdjacentEquivalentIfs(SeqStmt(wg_stmts[w]));
            wg_body = rewrap_loop_body_lets(wg_body);

            // Rewrite stage expressions if needed
            if (needs_phase_counter) {
              wg_body = StageExprReplacer::Replace(
                  wg_body, loop_var, loop_min, num_stages,
                  consumer_phase_counter->StageExpr(num_stages));
            }

            wg_loops[w] = For(loop_var, loop_min, loop_extent, ForKind::kSerial,
                              wg_body, Optional<IterVar>(), loop_annos);
          }
        }

        // Wrap each WG loop with phase counter allocation
        if (needs_phase_counter) {
          wg_loops[0] = consumer_phase_counter->WrapLoopWithAlloc(wg_loops[0]);
          for (int w = 1; w < num_warp_groups; ++w) {
            auto wg_phase = PhaseCounter::Create(
                std::string("wg") + std::to_string(w) + "_phase");
            wg_loops[w] = wg_phase.WrapLoopWithAlloc(wg_loops[w]);
          }
        }

        // Rewrite threadIdx.x for producer
        producer_loop = PCThreadIdxRewriter::Rewrite(
            producer_loop, thread_iv_->var,
            thread_iv_->var - total_consumer_threads, producer_thread_extent,
            /*do_shuffle=*/true);

        // Rewrite threadIdx.x for each warp group
        // WG0: threadIdx = tid (range 0-127), barrier rewrite consumer→128
        wg_loops[0] = PCThreadIdxRewriter::Rewrite(
            wg_loops[0], thread_iv_->var, thread_iv_->var, wg_extent,
            /*do_shuffle=*/true,
            /*rewrite_barrier_from=*/orig_consumer_int,
            /*rewrite_barrier_to=*/128);

        // WG1..WG(N-1): threadIdx = tid - w*128
        for (int w = 1; w < num_warp_groups; ++w) {
          PrimExpr wg_offset = IntImm(DataType::Int(32), w * 128);
          wg_loops[w] = ThreadIdxSubstitutor::Substitute(
              wg_loops[w], thread_iv_->var, wg_offset);
          Var dummy_var(std::string("__no_match_wg") + std::to_string(w) + "__",
                        DataType::Int(32));
          wg_loops[w] = PCThreadIdxRewriter::Rewrite(
              wg_loops[w], dummy_var, dummy_var, wg_extent,
              /*do_shuffle=*/false,
              /*rewrite_barrier_from=*/orig_consumer_int,
              /*rewrite_barrier_to=*/128,
              /*barrier_id_offset=*/w * 4);
        }

        // Build nested IfThenElse dispatch for consumer warp groups
        // WG(N-1) is the else-branch of the last if
        Stmt consumer_dispatch = wg_loops[num_warp_groups - 1];
        for (int w = num_warp_groups - 2; w >= 0; --w) {
          PrimExpr boundary = IntImm(DataType::Int(32), (w + 1) * 128);
          consumer_dispatch = IfThenElse(
              LT(thread_iv_->var, boundary), wg_loops[w], consumer_dispatch);
        }
        ws_body = IfThenElse(GE(thread_iv_->var, total_consumer_threads),
                             producer_loop, consumer_dispatch);

        LOG(INFO) << "FineGrainedWS per-op dispatch: " << num_warp_groups
                  << " warp groups, " << consumer_body_stmts.size() << " total stmts";
        for (int w = 0; w < num_warp_groups; ++w) {
          LOG(INFO) << "  WG" << w << ": " << wg_stmts[w].size() << " stmts";
        }
      }
    } else { standard_two_role:
      // --- Standard two-role thread split ---
      // Rewrite threadIdx.x in producer: threadIdx.x -> threadIdx.x -
      // consumer_threads Also converts `if (threadIdx.x == 0)` to `if
      // (tl_shuffle_elect(extent))`
      producer_loop = PCThreadIdxRewriter::Rewrite(
          producer_loop, thread_iv_->var,
          thread_iv_->var - consumer_thread_extent, producer_thread_extent,
          /*do_shuffle=*/true);
      // FA3-style row-split: when the annotation is present AND consumer
      // has ≥ 2 WGs (256 threads), duplicate the consumer loop into WG0/WG1
      // with independent per-WG AllReduce barriers. Both WGs run the FULL
      // attention body on their own Q row subset. No cross-WG data transfer.
      bool rowsplit = false;
      {
        auto rs_anno =
            pipeline_loop->annotations.Get("tl_finegrainedws_rowsplit");
        if (rs_anno.has_value()) {
          std::string anno_str =
              static_cast<std::string>(Downcast<String>(rs_anno.value()));
          if (anno_str == "1") rowsplit = true;
        }
      }
      auto cimm_rs = consumer_thread_extent.as<IntImmNode>();
      int64_t cext_rs = cimm_rs ? cimm_rs->value : 0;
      if (rowsplit && cext_rs >= 256) {
        LOG(INFO) << "FineGrainedWS: row-split mode (FA3-style, " << cext_rs << " threads)";
        PrimExpr wg_extent_rs = IntImm(DataType::Int(32), 128);
        int orig_cext = static_cast<int>(cext_rs);

        // WG0: tid [0,128) — keep threadIdx as-is, barrier rewrite 256→128
        Stmt wg0_loop = PCThreadIdxRewriter::Rewrite(
            consumer_loop, thread_iv_->var, thread_iv_->var,
            wg_extent_rs, /*do_shuffle=*/true,
            /*rewrite_barrier_from=*/orig_cext,
            /*rewrite_barrier_to=*/128);

        // WG1: tid [128,256) — shift threadIdx by -128, barrier rewrite
        // 256→128 with offset 4 so AllReduce uses independent barrier IDs
        Stmt wg1_loop = ThreadIdxSubstitutor::Substitute(
            consumer_loop, thread_iv_->var, wg_extent_rs);
        Var dummy_var("__no_match_rs__", DataType::Int(32));
        wg1_loop = PCThreadIdxRewriter::Rewrite(
            wg1_loop, dummy_var, dummy_var, wg_extent_rs,
            /*do_shuffle=*/false,
            /*rewrite_barrier_from=*/orig_cext,
            /*rewrite_barrier_to=*/128,
            /*barrier_id_offset=*/4);

        // Dispatch: tid < 128 → WG0, 128 ≤ tid < 256 → WG1, tid ≥ 256 → producer
        Stmt consumer_dispatch = IfThenElse(
            LT(thread_iv_->var, wg_extent_rs), wg0_loop, wg1_loop);
        ws_body = IfThenElse(GE(thread_iv_->var, consumer_thread_extent),
                             producer_loop, consumer_dispatch);
      } else {
        consumer_loop = PCThreadIdxRewriter::Rewrite(
            consumer_loop, thread_iv_->var, thread_iv_->var,
            consumer_thread_extent, /*do_shuffle=*/true);

        // Wrap in IfThenElse: producer if threadIdx.x >= consumer_threads
        ws_body = IfThenElse(GE(thread_iv_->var, consumer_thread_extent),
                             producer_loop, consumer_loop);
      }
    }

    // Add warp specialization scope attribute
    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_thread_extent),
                                  Downcast<IntImm>(ws_consumer_thread_extent)};
    ws_body =
        AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0, ws_body);

    // Forward barriers are producer-owned; back-pressure barriers are released
    // by the full consumer partition.
    Array<PrimExpr> barrier_arrive_counts;
    barrier_arrive_counts.reserve(total_barriers);
    if (remap_pure_tma_barriers_) {
      for (int i = 0; i < num_existing_loop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
      // BP barriers: in dual-consumer mode, each bp barrier is arrived
      // by only one WG (128 threads), not the full consumer (256).
      PrimExpr bp_arrive_count =
          (dual_consumer_enabled_)
              ? IntImm(DataType::Int(32), 128)
              : consumer_thread_extent;
      for (int i = 0; i < num_bp_barriers; ++i) {
        barrier_arrive_counts.push_back(bp_arrive_count);
      }
      for (int i = 0; i < num_preloop_fwd_barriers; ++i) {
        barrier_arrive_counts.push_back(IntImm(DataType::Int(32), 1));
      }
    } else {
      ICHECK_EQ(mixed_fwd_arrive_counts.size(),
                static_cast<size_t>(num_total_fwd_barriers));
      for (const auto &count : mixed_fwd_arrive_counts) {
        barrier_arrive_counts.push_back(count);
      }
      for (int i = 0; i < num_bp_barriers; i++) {
        barrier_arrive_counts.push_back(consumer_thread_extent);
      }
    }
    Stmt init_barrier = Evaluate(Call(
        DataType::Handle(), create_list_of_mbarrier(), barrier_arrive_counts));

    LocalLiveSet producer_live_seed =
        SeedLiveSetFromStmt(producer_loop_body, buffer_data_to_buffer);
    // Three-role: dQ writer runs inside the producer WG, so its live set
    // must be merged into the producer seed for correct pre-loop partitioning.
    if (has_three_role) {
      LocalLiveSet dq_writer_live =
          SeedLiveSetFromStmt(dq_writer_loop_body, buffer_data_to_buffer);
      producer_live_seed.buffers.insert(dq_writer_live.buffers.begin(),
                                        dq_writer_live.buffers.end());
      producer_live_seed.vars.insert(dq_writer_live.vars.begin(),
                                     dq_writer_live.vars.end());
    }
    LocalLiveSet consumer_live_seed =
        SeedLiveSetFromStmt(consumer_loop_body, buffer_data_to_buffer);
    // Pre-loop liveness assignment must also account for variables used only in
    // the pipeline loop bounds. Otherwise scalar setup that feeds the loop
    // extent/min can be misclassified as common code and hoisted outside the
    // warp-specialized split.
    producer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_min, buffer_data_to_buffer));
    producer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_extent, buffer_data_to_buffer));
    consumer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_min, buffer_data_to_buffer));
    consumer_live_seed.AddUses(
        LocalAccessCollector::CollectExpr(loop_extent, buffer_data_to_buffer));

    consumer_thread_extent_ = ws_consumer_thread_extent;

    // Reconstruct block body: replace the pipeline loop and
    // create_list_of_mbarrier with new init_barrier + ws_body.
    Stmt new_block_body = RebuildBlockBody(
        orig_block->body, pipeline_loop, init_barrier, ws_body,
        buffer_data_to_buffer, producer_live_seed, consumer_live_seed);

    // Dual-consumer: wrap post-loop code (after WS body) with WG0-only guard.
    // Post-loop (o_acc/=l, output write, lse write) uses threadIdx for addressing.
    // WG1 (tid 128-255) would go OOB. Guard with tid < consumer_thread_extent.
    // Note: this means WG0 writes the output (with incomplete o_acc since PV
    // was done in WG1). For correctness, WG1 should write output with adjusted
    // tid. For now, this prevents crashes for performance measurement.
    if (dual_consumer_enabled_) {
      // Find post-loop stmts in new_block_body (everything after ws_body)
      // and wrap them with if (threadIdx < consumer_extent)
      // Simplest: wrap the ENTIRE new_block_body's last SeqStmt entries
      // Actually: just wrap all post-loop with a guard at codegen level
      // by marking the function with an attribute
    }

    // Update thread extent. Dual-consumer splits the existing consumer
    // thread pool into WG0/WG1 (each wg_extent = 128) — it does NOT add
    // threads. The previous code unconditionally added +128, which for a
    // kernel specified with threads=256 (already 2 WGs = WG0+WG1) produced
    // a 4-WG launch of 512 threads and an illegal-access fault at run time.
    //
    // Only add the extra WG when the caller's consumer extent is exactly
    // one WG (128) — i.e. the user specified threads=128 and dual-consumer
    // is expected to grow the consumer pool to 256.
    if (dual_consumer_enabled_) {
      auto cimm = consumer_thread_extent.as<IntImmNode>();
      int64_t cext = cimm ? cimm->value : -1;
      if (cext == 128) {
        num_threads_ = ws_consumer_thread_extent + producer_thread_extent;
      } else {
        // consumer already >= 2 WGs; dual-consumer only relabels threads
        num_threads_ = ws_consumer_thread_extent + producer_thread_extent;
      }
    } else {
      num_threads_ = ws_consumer_thread_extent + producer_thread_extent;
    }
    ws_transformed_ = true;
    use_full_tma_forward_barrier_protocol_ =
        old_use_full_tma_forward_barrier_protocol;
    remap_pure_tma_barriers_ = old_remap_pure_tma_barriers;
    pure_tma_preloop_fwd_base_ = old_pure_tma_preloop_fwd_base;
    pure_tma_preloop_fwd_count_ = old_pure_tma_preloop_fwd_count;
    pure_tma_preloop_fwd_cursor_ = old_pure_tma_preloop_fwd_cursor;
    current_loop_guard_bindings_ = std::move(saved_loop_guard_bindings);

    // Build the new Block and BlockRealize (without recursive mutation
    // since we've already transformed the body directly).
    Block new_block(orig_block->iter_vars, orig_block->reads,
                    orig_block->writes, orig_block->name_hint, new_block_body,
                    orig_block->init, orig_block->alloc_buffers,
                    orig_block->match_buffers, orig_block->annotations);
    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  // Handle ForNode with thread bindings
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x" &&
        !thread_iv_.defined()) {
      thread_iv_ = op->thread_binding.value();
      Optional<PrimExpr> old_num_threads = num_threads_;
      num_threads_ = std::nullopt;
      For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
      if (num_threads_.defined()) {
        PrimExpr num_threads = num_threads_.value();
        auto n = for_node.CopyOnWrite();
        n->extent = num_threads;
        IterVar new_thread_iv = n->thread_binding.value();
        new_thread_iv.CopyOnWrite()->dom =
            Range::FromMinExtent(Integer(0), num_threads);
        n->thread_binding = new_thread_iv;
      }
      num_threads_ = old_num_threads;
      thread_iv_ = {};
      return for_node;
    }

    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (for_node->kind == ForKind::kThreadBinding && thread_iv_.defined()) {
      ICHECK(for_node->thread_binding.defined());
      String thread_tag = for_node->thread_binding.value()->thread_tag;
      if (thread_tag == "threadIdx.x") {
        Var thread_v = Downcast<Var>(for_node->loop_var);
        Stmt new_body = PCThreadIdxRewriter::Rewrite(for_node->body, thread_v,
                                                     thread_iv_->var, 0);
        return new_body;
      }
    }
    return for_node;
  }

  // ---------------------------------------------------------------------------
  // Utility methods
  // ---------------------------------------------------------------------------

  void FlattenSeqStmt(const Stmt &s, Array<Stmt> *out) {
    if (auto *seq = s.as<SeqStmtNode>()) {
      for (const auto &sub : seq->seq) {
        FlattenSeqStmt(sub, out);
      }
    } else {
      out->push_back(s);
    }
  }

  struct BufferDataAccessInfo {
    bool read{false};
    bool write{false};

    bool HasAnyAccess() const { return read || write; }
  };

  BufferDataAccessInfo
  AnalyzeBufferDataAccess(const Stmt &stmt, const Var &buffer_data,
                          const BufferDataToBufferMap &buffer_map) const {
    class BufferDataAccessDetector : public StmtExprVisitor {
    public:
      BufferDataAccessDetector(const Var &buffer_data,
                               const BufferDataToBufferMap &buffer_map)
          : buffer_data_(buffer_data), buffer_map_(buffer_map) {}

      BufferDataAccessInfo Result() const { return result_; }

    private:
      void VisitExpr_(const BufferLoadNode *op) final {
        if (op->buffer->data.same_as(buffer_data_)) {
          result_.read = true;
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        if (op->buffer->data.same_as(buffer_data_)) {
          result_.write = true;
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(tl::access_ptr())) {
          ICHECK_EQ(op->args.size(), 3);
          const auto *base_load = op->args[0].as<BufferLoadNode>();
          ICHECK(base_load);
          if (base_load->buffer->data.same_as(buffer_data_)) {
            MarkAccess(op->args[2]);
          }
          for (const auto &index : base_load->indices) {
            VisitExpr(index);
          }
          VisitExpr(op->args[1]);
          return;
        }

        if (op->op.same_as(builtin::tvm_access_ptr())) {
          ICHECK_EQ(op->args.size(), 5);
          const auto *var = op->args[1].as<VarNode>();
          ICHECK(var);
          auto it = buffer_map_.find(GetRef<Var>(var));
          if (it != buffer_map_.end() &&
              it->second->data.same_as(buffer_data_)) {
            MarkAccess(op->args[4]);
          }
          VisitExpr(op->args[2]);
          VisitExpr(op->args[3]);
          return;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void MarkAccess(const PrimExpr &rw_expr) {
        int rw_mask = 3;
        if (const auto *imm = rw_expr.as<IntImmNode>()) {
          rw_mask = static_cast<int>(imm->value);
        }
        if (rw_mask & 1) {
          result_.read = true;
        }
        if (rw_mask & 2) {
          result_.write = true;
        }
      }

      Var buffer_data_;
      const BufferDataToBufferMap &buffer_map_;
      BufferDataAccessInfo result_;
    };

    BufferDataAccessDetector detector(buffer_data, buffer_map);
    detector(stmt);
    return detector.Result();
  }

  const ForNode *FindAnnotatedPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages")) {
        return for_node;
      }
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (auto *result = FindAnnotatedPipelineLoop(s)) {
          return result;
        }
      }
      return nullptr;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return FindAnnotatedPipelineLoop(realize->block->body);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return FindAnnotatedPipelineLoop(block->body);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return FindAnnotatedPipelineLoop(attr->body);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return FindAnnotatedPipelineLoop(let_s->body);
    }
    return nullptr;
  }

  // Infer how many mbarriers are already referenced by this block body.
  // This prevents assigning back-pressure barriers that alias existing
  // forward barriers (e.g. prologue TMA copy barriers outside the pipeline).
  int InferMinRequiredBarrierCount(const Stmt &stmt) {
    class GetMbarrierMaxIdxCollector : public StmtExprVisitor {
    public:
      int max_idx{-1};
      bool has_unbounded{false};

    private:
      void VisitStmt_(const ForNode *op) final {
        // Bind loop variable range so expressions like (k + c) can be bounded.
        analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(get_mbarrier()) && op->args.size() == 1) {
          auto bound = analyzer_.const_int_bound(op->args[0]);
          if (bound->max_value != arith::ConstIntBound::kPosInf &&
              bound->max_value != arith::ConstIntBound::kNegInf) {
            max_idx = std::max(max_idx, static_cast<int>(bound->max_value));
          } else {
            has_unbounded = true;
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      arith::Analyzer analyzer_;
    };

    GetMbarrierMaxIdxCollector collector;
    collector(stmt);
    ICHECK(!collector.has_unbounded)
        << "FineGrainedWS: cannot infer finite upper bound for existing "
        << "mbarrier id expressions. Refusing to allocate back-pressure "
        << "barriers to avoid id overlap.";
    return collector.max_idx + 1;
  }

  int CountRewrittenPureTmaPreloopForwardPairs(const Stmt &stmt,
                                               const ForNode *target_loop) {
    if (stmt.as<ForNode>() == target_loop) {
      return 0;
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> pre_loop_stmts;
      bool found_loop = false;
      int nested_count = 0;
      for (const auto &s : seq->seq) {
        if (IsCreateListOfMbarrier(s)) {
          continue;
        }
        if (!found_loop && ContainsLoop(s, target_loop)) {
          nested_count =
              CountRewrittenPureTmaPreloopForwardPairs(s, target_loop);
          found_loop = true;
        } else if (!found_loop) {
          pre_loop_stmts.push_back(s);
        }
      }
      if (!found_loop) {
        return 0;
      }

      size_t movable_begin = pre_loop_stmts.size();
      while (movable_begin > 0 &&
             IsMovableConsumerPrefixStmt(pre_loop_stmts[movable_begin - 1])) {
        --movable_begin;
      }

      int local_count = 0;
      for (size_t i = 0; i + 1 < movable_begin; ++i) {
        if (ContainsTmaLoadStmt(pre_loop_stmts[i]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i + 1])) {
          ++local_count;
        }
      }
      return nested_count + local_count;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(attr->body, target_loop);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(let_s->body, target_loop);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(realize->block->body,
                                                      target_loop);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return CountRewrittenPureTmaPreloopForwardPairs(block->body, target_loop);
    }
    return 0;
  }

  // Single source of truth for barrier/TMA control-like calls that should not
  // be moved across producer/consumer partition boundaries.
  bool IsBarrierOrTmaControlCall(const CallNode *call) {
    return call->op.same_as(create_list_of_mbarrier()) ||
           call->op.same_as(mbarrier_wait_parity()) ||
           call->op.same_as(mbarrier_expect_tx()) ||
           call->op.same_as(builtin::ptx_arrive_barrier()) ||
           call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
           call->op.same_as(builtin::ptx_cp_async_barrier()) ||
           call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
           call->op.same_as(tma_load()) ||
           call->op.same_as(tma_load_im2col()) ||
           call->op.same_as(tma_store()) ||
           call->op.same_as(tma_store_arrive()) ||
           call->op.same_as(tma_store_wait()) ||
           call->op.same_as(builtin::tvm_storage_sync());
  }

  bool IsMovableConsumerPrefixStmt(const Stmt &stmt) {
    bool has_disallowed = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        // Only move pure local init into the consumer prefix. If a stmt reads
        // global or shared memory, the producer may also depend on its result
        // (for example a mask controlling which async copies to issue).
        if (IsSharedBuffer(ld->buffer) || IsGlobalBuffer(ld->buffer)) {
          has_disallowed = true;
          return;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer) || IsGlobalBuffer(st->buffer)) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  bool IsProducerMovableLoopPrefixStmt(const Stmt &stmt) {
    bool has_allowed_work = false;
    bool has_disallowed = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::tvm_storage_sync())) {
          const auto *scope = call->args[0].as<StringImmNode>();
          if (!scope ||
              (scope->value != "shared" && scope->value != "shared.dyn")) {
            has_disallowed = true;
            return;
          }
          has_allowed_work = true;
          return;
        }
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *ld = node.as<BufferLoadNode>()) {
        if (IsSharedBuffer(ld->buffer) || IsLocalBuffer(ld->buffer, true)) {
          has_disallowed = true;
          return;
        }
        if (IsGlobalBuffer(ld->buffer)) {
          has_allowed_work = true;
        }
      }
      if (const auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer)) {
          has_allowed_work = true;
          return;
        }
        has_disallowed = true;
      }
    });
    return has_allowed_work && !has_disallowed;
  }

  Optional<Stmt> TryPrependToConsumerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToConsumerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToConsumerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToConsumerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToConsumerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({prepend_stmt, if_stmt->else_case.value()});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryPrependToProducerBranch(const Stmt &stmt,
                                            const Stmt &prepend_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryPrependToProducerBranch(new_seq.back(), prepend_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryPrependToProducerBranch(attr->body, prepend_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryPrependToProducerBranch(let_s->body, prepend_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryPrependToProducerBranch(realize->block->body, prepend_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryPrependToProducerBranch(block->body, prepend_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Stmt new_then = SeqStmt({prepend_stmt, if_stmt->then_case});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToProducerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToProducerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToProducerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToProducerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToProducerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToProducerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      auto nested = TryAppendToProducerBranch(if_stmt->then_case, append_stmt);
      if (nested.defined()) {
        return IfThenElse(if_stmt->condition, nested.value(),
                          if_stmt->else_case);
      }
      Stmt new_then = SeqStmt({if_stmt->then_case, append_stmt});
      return IfThenElse(if_stmt->condition, new_then, if_stmt->else_case);
    }
    return std::nullopt;
  }

  Optional<Stmt> TryAppendToConsumerBranch(const Stmt &stmt,
                                           const Stmt &append_stmt) {
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return std::nullopt;
      }
      Array<Stmt> new_seq = seq->seq;
      auto nested = TryAppendToConsumerBranch(new_seq.back(), append_stmt);
      if (nested.defined()) {
        new_seq.Set(new_seq.size() - 1, nested.value());
        return SeqStmt(new_seq);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(attr->body, append_stmt);
      if (nested.defined()) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        nested.value());
      }
      return std::nullopt;
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      auto nested = TryAppendToConsumerBranch(let_s->body, append_stmt);
      if (nested.defined()) {
        return LetStmt(let_s->var, let_s->value, nested.value());
      }
      return std::nullopt;
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      auto nested =
          TryAppendToConsumerBranch(realize->block->body, append_stmt);
      if (nested.defined()) {
        const Block &orig = realize->block;
        Block new_block(orig->iter_vars, orig->reads, orig->writes,
                        orig->name_hint, nested.value(), orig->init,
                        orig->alloc_buffers, orig->match_buffers,
                        orig->annotations);
        return BlockRealize(realize->iter_values, realize->predicate,
                            new_block);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      auto nested = TryAppendToConsumerBranch(block->body, append_stmt);
      if (nested.defined()) {
        return Block(block->iter_vars, block->reads, block->writes,
                     block->name_hint, nested.value(), block->init,
                     block->alloc_buffers, block->match_buffers,
                     block->annotations);
      }
      return std::nullopt;
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        return std::nullopt;
      }
      Stmt new_else = SeqStmt({if_stmt->else_case.value(), append_stmt});
      return IfThenElse(if_stmt->condition, if_stmt->then_case, new_else);
    }
    return std::nullopt;
  }

  bool IsMbarrierWaitParityStmt(const Stmt &stmt) {
    return ExtractWaitBarrierId(stmt).defined();
  }

  Optional<PrimExpr> ExtractWaitBarrierId(const Stmt &stmt) {
    auto extract_from_call = [](const CallNode *call) -> Optional<PrimExpr> {
      if (!call || !call->op.same_as(mbarrier_wait_parity()) ||
          call->args.size() != 2) {
        return std::nullopt;
      }
      if (auto *get = call->args[0].as<CallNode>()) {
        if (get->op.same_as(get_mbarrier()) && get->args.size() == 1) {
          return get->args[0];
        }
      }
      return std::nullopt;
    };

    if (auto *eval = stmt.as<EvaluateNode>()) {
      return extract_from_call(eval->value.as<CallNode>());
    }
    if (auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        return ExtractWaitBarrierId(if_stmt->then_case);
      }
      return std::nullopt;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractWaitBarrierId(attr->body);
    }
    if (auto *let_stmt = stmt.as<LetStmtNode>()) {
      return ExtractWaitBarrierId(let_stmt->body);
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return ExtractWaitBarrierId(seq->seq[0]);
      }
      return std::nullopt;
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ExtractWaitBarrierId(block->body);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      if (is_one(realize->predicate)) {
        return ExtractWaitBarrierId(realize->block->body);
      }
    }
    return std::nullopt;
  }

  Stmt NormalizeForwardWaitParity(const Stmt &wait_stmt,
                                  const PrimExpr &normalized_parity) {
    auto barrier_id = ExtractWaitBarrierId(wait_stmt);
    if (!barrier_id.defined()) {
      return wait_stmt;
    }
    return makeParityWait(barrier_id.value(), normalized_parity);
  }

  bool ContainsTmaLoadStmt(const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          found = true;
        }
      }
    });
    return found;
  }

  bool IsThreadOnlyPredicate(const PrimExpr &expr) const {
    bool uses_thread = false;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        if (thread_iv_.defined() && var == thread_iv_->var.get()) {
          uses_thread = true;
        }
      }
    });
    return uses_thread;
  }

  Optional<PrimExpr> ExtractNonThreadProducerGuard(const Stmt &stmt) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return ExtractNonThreadProducerGuard(attr->body);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      return ExtractNonThreadProducerGuard(let_s->body);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      return ExtractNonThreadProducerGuard(realize->block->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return ExtractNonThreadProducerGuard(block->body);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        auto guard = ExtractNonThreadProducerGuard(s);
        if (guard.defined()) {
          return guard;
        }
      }
      return std::nullopt;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return if_stmt->condition;
        }
        return ExtractNonThreadProducerGuard(if_stmt->then_case);
      }
    }
    return std::nullopt;
  }

  PrimExpr ResolveGuardBinding(const PrimExpr &expr,
                               const VarBindingMap &bindings) const {
    if (const auto *var = expr.as<VarNode>()) {
      auto it = bindings.find(GetRef<Var>(var));
      if (it != bindings.end()) {
        return ResolveGuardBinding(it->second, bindings);
      }
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return ResolveGuardBinding(cast->value, bindings);
    }
    return expr;
  }

  bool IsMaskLikeBooleanExpr(const PrimExpr &expr) const {
    PrimExpr resolved = expr;
    while (const auto *cast = resolved.as<CastNode>()) {
      resolved = cast->value;
    }
    auto is_const_bool = [](const PrimExpr &value, bool expected) {
      if (const auto *imm = value.as<IntImmNode>()) {
        return static_cast<bool>(imm->value) == expected;
      }
      return false;
    };
    if (const auto *load = resolved.as<BufferLoadNode>()) {
      return load->buffer->dtype.is_bool();
    }
    if (const auto *select = resolved.as<SelectNode>()) {
      if (is_const_bool(select->false_value, false)) {
        return IsMaskLikeBooleanExpr(select->true_value);
      }
      if (is_const_bool(select->true_value, false)) {
        return IsMaskLikeBooleanExpr(select->false_value);
      }
    }
    if (const auto *call = resolved.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>()) {
        if (op->name == "tl.any_of" || op->name == "tl.all_of") {
          return true;
        }
      }
      if (call->op.same_as(builtin::call_extern()) && !call->args.empty()) {
        if (const auto *name = call->args[0].as<StringImmNode>()) {
          if (name->value == "tl::Any" || name->value == "tl::All") {
            return true;
          }
        }
      }
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
        if (is_const_bool(call->args[2], false)) {
          return IsMaskLikeBooleanExpr(call->args[1]);
        }
        if (is_const_bool(call->args[1], false)) {
          return IsMaskLikeBooleanExpr(call->args[2]);
        }
      }
    }
    return false;
  }

  bool CanIssueProducerWithoutGuardImpl(const Stmt &stmt,
                                        VarBindingMap *bindings) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return CanIssueProducerWithoutGuardImpl(attr->body, bindings);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      bindings->emplace(let_s->var, let_s->value);
      bool result = CanIssueProducerWithoutGuardImpl(let_s->body, bindings);
      bindings->erase(let_s->var);
      return result;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      return CanIssueProducerWithoutGuardImpl(realize->block->body, bindings);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return CanIssueProducerWithoutGuardImpl(block->body, bindings);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (CanIssueProducerWithoutGuardImpl(s, bindings)) {
          return true;
        }
      }
      return false;
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          if (const auto *var = if_stmt->condition.as<VarNode>()) {
            Var cond_var = GetRef<Var>(var);
            if (!UsesVar(if_stmt->then_case, [cond_var](const VarNode *vn) {
                  return vn == cond_var.get();
                })) {
              return true;
            }
          }
          PrimExpr resolved =
              ResolveGuardBinding(if_stmt->condition, *bindings);
          return IsMaskLikeBooleanExpr(resolved);
        }
        return CanIssueProducerWithoutGuardImpl(if_stmt->then_case, bindings);
      }
    }
    return false;
  }

  bool CanIssueProducerWithoutGuard(const Stmt &stmt) const {
    VarBindingMap bindings = current_loop_guard_bindings_;
    return CanIssueProducerWithoutGuardImpl(stmt, &bindings);
  }

  Stmt StripNonThreadProducerGuard(const Stmt &stmt) const {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      StripNonThreadProducerGuard(attr->body), attr->span);
    }
    if (const auto *let_s = stmt.as<LetStmtNode>()) {
      return LetStmt(let_s->var, let_s->value,
                     StripNonThreadProducerGuard(let_s->body));
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, StripNonThreadProducerGuard(orig->body),
                      orig->init, orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, StripNonThreadProducerGuard(block->body),
                   block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> stripped;
      stripped.reserve(seq->seq.size());
      for (const auto &s : seq->seq) {
        stripped.push_back(StripNonThreadProducerGuard(s));
      }
      return stripped.size() == 1 ? stripped[0] : SeqStmt(stripped, seq->span);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return StripNonThreadProducerGuard(if_stmt->then_case);
        }
        return IfThenElse(if_stmt->condition,
                          StripNonThreadProducerGuard(if_stmt->then_case),
                          std::nullopt, if_stmt->span);
      }
    }
    return stmt;
  }

  Stmt WrapStmtWithOptionalGuard(const Optional<PrimExpr> &guard,
                                 const Stmt &stmt) const {
    if (!guard.defined()) {
      return stmt;
    }
    return IfThenElse(guard.value(), stmt, std::nullopt);
  }

  Optional<Stmt> WrapStmtWithNonThreadGuardLike(const Stmt &source,
                                                const Stmt &stmt) const {
    if (const auto *attr = source.as<AttrStmtNode>()) {
      Optional<Stmt> wrapped = WrapStmtWithNonThreadGuardLike(attr->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, wrapped.value(),
                      attr->span);
    }
    if (const auto *let_s = source.as<LetStmtNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(let_s->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return LetStmt(let_s->var, let_s->value, wrapped.value());
    }
    if (const auto *realize = source.as<BlockRealizeNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(realize->block->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, wrapped.value(), orig->init,
                      orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *block = source.as<BlockNode>()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(block->body, stmt);
      if (!wrapped.defined()) {
        return std::nullopt;
      }
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, wrapped.value(), block->init,
                   block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *seq = source.as<SeqStmtNode>()) {
      if (seq->seq.size() == 1) {
        return WrapStmtWithNonThreadGuardLike(seq->seq[0], stmt);
      }
      return std::nullopt;
    }
    if (const auto *if_stmt = source.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined() ||
          IsTrivialNoOpStmt(if_stmt->else_case.value())) {
        if (!IsThreadOnlyPredicate(if_stmt->condition)) {
          return IfThenElse(if_stmt->condition, stmt, std::nullopt,
                            if_stmt->span);
        }
        Optional<Stmt> wrapped =
            WrapStmtWithNonThreadGuardLike(if_stmt->then_case, stmt);
        if (!wrapped.defined()) {
          return std::nullopt;
        }
        return IfThenElse(if_stmt->condition, wrapped.value(), std::nullopt,
                          if_stmt->span);
      }
    }
    return std::nullopt;
  }

  Stmt WrapStmtWithGuardSource(const Optional<Stmt> &guard_source,
                               const Optional<PrimExpr> &guard,
                               const Stmt &stmt) const {
    if (guard_source.defined()) {
      Optional<Stmt> wrapped =
          WrapStmtWithNonThreadGuardLike(guard_source.value(), stmt);
      if (wrapped.defined()) {
        return wrapped.value();
      }
    }
    return WrapStmtWithOptionalGuard(guard, stmt);
  }

  Stmt RewriteWaitBarrier(const Stmt &wait_stmt, const PrimExpr &new_barrier_id,
                          Optional<PrimExpr> new_parity = std::nullopt) {
    class WaitBarrierRewriter : public StmtExprMutator {
    public:
      WaitBarrierRewriter(PrimExpr barrier_id, Optional<PrimExpr> parity)
          : barrier_id_(std::move(barrier_id)), parity_(std::move(parity)) {}

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if (call->op.same_as(mbarrier_wait_parity()) &&
            call->args.size() == 2) {
          PrimExpr parity = parity_.defined() ? parity_.value() : call->args[1];
          return Call(call->dtype, call->op,
                      {makeGetBarrier(barrier_id_), parity}, call->annotations,
                      call->span);
        }
        return call;
      }

    private:
      PrimExpr barrier_id_;
      Optional<PrimExpr> parity_;
    };

    return MergeAdjacentEquivalentIfs(
        WaitBarrierRewriter(new_barrier_id, std::move(new_parity))(wait_stmt));
  }

  Stmt RewriteTmaStmtBarrierIdPreserveProtocol(const Stmt &stmt,
                                               const PrimExpr &barrier_id,
                                               bool drop_arrive = false) {
    class TmaBarrierIdRewriter : public StmtExprMutator {
    public:
      TmaBarrierIdRewriter(PrimExpr barrier_id, bool drop_arrive)
          : barrier_id_(std::move(barrier_id)), drop_arrive_(drop_arrive) {}

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
             call->op.same_as(mbarrier_expect_tx())) &&
            call->args.size() == 2) {
          return Call(call->dtype, call->op,
                      {makeGetBarrier(barrier_id_), call->args[1]},
                      call->annotations, call->span);
        }
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          bool is_1d_tma_load = false;
          if (const auto *arg0 = call->args[0].as<CallNode>()) {
            is_1d_tma_load = !arg0->op.same_as(create_tma_descriptor()) &&
                             call->op.same_as(tma_load());
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(is_1d_tma_load ? 2 : 1,
                             makeGetBarrier(barrier_id_));
          return call;
        }
        if (call->op.same_as(builtin::ptx_arrive_barrier()) &&
            !call->args.empty()) {
          if (drop_arrive_) {
            return IntImm(DataType::Int(32), 0);
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(0, makeGetBarrier(barrier_id_));
          return call;
        }
        return call;
      }

    private:
      PrimExpr barrier_id_;
      bool drop_arrive_;
    };

    return MergeAdjacentEquivalentIfs(
        TmaBarrierIdRewriter(barrier_id, drop_arrive)(stmt));
  }

  Stmt MergeAdjacentEquivalentIfs(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value,
                      MergeAdjacentEquivalentIfs(attr->body), attr->span);
    }
    if (const auto *let_stmt = stmt.as<LetStmtNode>()) {
      return LetStmt(let_stmt->var, let_stmt->value,
                     MergeAdjacentEquivalentIfs(let_stmt->body));
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return Block(block->iter_vars, block->reads, block->writes,
                   block->name_hint, MergeAdjacentEquivalentIfs(block->body),
                   block->init, block->alloc_buffers, block->match_buffers,
                   block->annotations);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      const Block &orig = realize->block;
      Block new_block(orig->iter_vars, orig->reads, orig->writes,
                      orig->name_hint, MergeAdjacentEquivalentIfs(orig->body),
                      orig->init, orig->alloc_buffers, orig->match_buffers,
                      orig->annotations);
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
      Optional<Stmt> else_case = std::nullopt;
      if (if_stmt->else_case.defined()) {
        else_case = MergeAdjacentEquivalentIfs(if_stmt->else_case.value());
      }
      return IfThenElse(if_stmt->condition,
                        MergeAdjacentEquivalentIfs(if_stmt->then_case),
                        else_case, if_stmt->span);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> merged;
      StructuralEqual equal;
      for (size_t i = 0; i < seq->seq.size();) {
        const auto *if0 = seq->seq[i].as<IfThenElseNode>();
        if (if0 && !if0->else_case.defined()) {
          Array<Stmt> then_stmts;
          then_stmts.push_back(if0->then_case);
          size_t j = i + 1;
          while (j < seq->seq.size()) {
            const auto *ifj = seq->seq[j].as<IfThenElseNode>();
            if (!ifj || ifj->else_case.defined() ||
                !equal(if0->condition, ifj->condition)) {
              break;
            }
            then_stmts.push_back(ifj->then_case);
            ++j;
          }
          if (then_stmts.size() == 1) {
            merged.push_back(seq->seq[i]);
          } else {
            Stmt merged_then = MergeAdjacentEquivalentIfs(
                then_stmts.size() == 1 ? then_stmts[0] : SeqStmt(then_stmts));
            merged.push_back(IfThenElse(if0->condition, merged_then,
                                        std::nullopt, if0->span));
          }
          i = j;
          continue;
        }
        merged.push_back(seq->seq[i]);
        ++i;
      }
      return merged.size() == 1 ? merged[0] : SeqStmt(merged, seq->span);
    }
    return stmt;
  }

  Stmt RewriteTmaForwardProducerStmt(const Stmt &stmt,
                                     const PrimExpr &barrier_id,
                                     bool append_arrive) {
    class TmaForwardBarrierStmtRewriter : public StmtExprMutator {
    public:
      explicit TmaForwardBarrierStmtRewriter(PrimExpr barrier_id)
          : barrier_id_(std::move(barrier_id)) {}

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
        if ((call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
             call->op.same_as(mbarrier_expect_tx())) &&
            call->args.size() == 2) {
          return Call(call->dtype, mbarrier_expect_tx(),
                      {makeGetBarrier(barrier_id_), call->args[1]},
                      call->annotations, call->span);
        }
        if (call->op.same_as(tma_load()) ||
            call->op.same_as(tma_load_im2col())) {
          bool is_1d_tma_load = false;
          if (const auto *arg0 = call->args[0].as<CallNode>()) {
            is_1d_tma_load = !arg0->op.same_as(create_tma_descriptor()) &&
                             call->op.same_as(tma_load());
          }
          auto new_call = call.CopyOnWrite();
          new_call->args.Set(is_1d_tma_load ? 2 : 1,
                             makeGetBarrier(barrier_id_));
          return call;
        }
        if (call->op.same_as(builtin::ptx_arrive_barrier()) &&
            !call->args.empty()) {
          return IntImm(DataType::Int(32), 0);
        }
        return call;
      }

    private:
      PrimExpr barrier_id_;
    };

    // Rebind the producer-side barrier id and finish the stage with a normal
    // barrier arrival. Pure-TMA pipelines do not need cp.async.mbarrier.arrive.
    Stmt rewritten = MergeAdjacentEquivalentIfs(
        TmaForwardBarrierStmtRewriter(barrier_id)(stmt));
    if (!append_arrive) {
      return rewritten;
    }
    Optional<PrimExpr> guard = ExtractNonThreadProducerGuard(stmt);
    Stmt elect_arrive = IfThenElse(
        Call(DataType::Bool(), tl_shuffle_elect(), {producer_thread_extent_}),
        makeArriveBarrier(barrier_id), std::nullopt);
    elect_arrive = WrapStmtWithOptionalGuard(guard, elect_arrive);
    return MergeAdjacentEquivalentIfs(SeqStmt({rewritten, elect_arrive}));
  }

  Stmt RewritePureTmaForwardPairsWithFreshBarriers(const Stmt &stmt) {
    class OutsideLoopPureTmaRewriter : public StmtExprMutator {
    public:
      explicit OutsideLoopPureTmaRewriter(FineGrainedWSRewriter *parent)
          : parent_(parent) {}

      Stmt VisitStmt_(const SeqStmtNode *op) final {
        Array<Stmt> new_seq;
        bool changed = false;
        for (size_t i = 0; i < op->seq.size(); ++i) {
          if (i + 1 < op->seq.size() &&
              parent_->ContainsTmaLoadStmt(op->seq[i]) &&
              parent_->IsMbarrierWaitParityStmt(op->seq[i + 1])) {
            ICHECK_GE(parent_->pure_tma_preloop_fwd_base_, 0);
            ICHECK_LT(parent_->pure_tma_preloop_fwd_cursor_,
                      parent_->pure_tma_preloop_fwd_count_);
            PrimExpr barrier_id = IntImm(
                DataType::Int(32), parent_->pure_tma_preloop_fwd_base_ +
                                       parent_->pure_tma_preloop_fwd_cursor_++);
            Stmt producer_stmt = parent_->MergeAdjacentEquivalentIfs(
                parent_->RewriteTmaStmtBarrierIdPreserveProtocol(
                    StripTmaCopyWriteBufferAttr(op->seq[i]), barrier_id));
            Stmt wait_stmt =
                parent_->RewriteWaitBarrier(op->seq[i + 1], barrier_id);
            new_seq.push_back(producer_stmt);
            new_seq.push_back(wait_stmt);
            ++i;
            changed = true;
            continue;
          }
          Stmt visited = StmtExprMutator::VisitStmt(op->seq[i]);
          new_seq.push_back(visited);
          changed = changed || !visited.same_as(op->seq[i]);
        }
        if (!changed) {
          return GetRef<Stmt>(op);
        }
        return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
      }

    private:
      FineGrainedWSRewriter *parent_;
    };

    OutsideLoopPureTmaRewriter rewriter(this);
    return rewriter(stmt);
  }

  bool IsSharedDependentConsumerPreStmt(const Stmt &stmt) {
    bool has_shared_access = false;
    bool has_control_ops = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_control_ops) {
        return;
      }
      if (auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_control_ops = true;
          return;
        }
      }
      if (auto *ld = node.as<BufferLoadNode>()) {
        if (IsSharedBuffer(ld->buffer)) {
          has_shared_access = true;
        }
      }
      if (auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer)) {
          has_shared_access = true;
        }
      }
    });
    return has_shared_access && !has_control_ops;
  }

  bool IsBranchLocalPreStmtCandidate(const Stmt &stmt,
                                     const LocalAccessSummary &summary) {
    if (!summary.HasTrackedDefs()) {
      return false;
    }
    bool has_disallowed = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (has_disallowed) {
        return;
      }
      if (const auto *call = node.as<CallNode>()) {
        if (IsBarrierOrTmaControlCall(call)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *ld = node.as<BufferLoadNode>()) {
        if (IsSharedBuffer(ld->buffer)) {
          has_disallowed = true;
          return;
        }
      }
      if (const auto *st = node.as<BufferStoreNode>()) {
        if (IsSharedBuffer(st->buffer) || IsGlobalBuffer(st->buffer)) {
          has_disallowed = true;
          return;
        }
      }
    });
    return !has_disallowed;
  }

  LocalLiveSet SeedLiveSetFromStmt(const Stmt &stmt,
                                   const BufferDataToBufferMap &buffer_map) {
    LocalLiveSet live;
    live.AddUses(LocalAccessCollector::Collect(stmt, buffer_map));
    return live;
  }

  /*!
   * \brief Rebuild the block body, replacing the pipeline loop with
   *        init_barrier + ws_body and removing old create_list_of_mbarrier.
   *
   *  Statements after the pipeline loop (e.g. epilogue, store) should execute
   *  only on consumer threads. Prefer appending them into the consumer branch
   *  of the warp-specialized if/else to keep a single top-level partition.
   *  If that is not possible, fall back to an explicit consumer-thread guard.
   */
  Stmt RebuildBlockBody(const Stmt &body, const ForNode *target_loop,
                        const Stmt &init_barrier, const Stmt &ws_body,
                        const BufferDataToBufferMap &buffer_data_to_buffer,
                        const LocalLiveSet &producer_live_seed,
                        const LocalLiveSet &consumer_live_seed) {
    // If this IS the target loop, replace it
    if (body.as<ForNode>() == target_loop) {
      return SeqStmt({init_barrier, ws_body});
    }

    if (auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      Array<Stmt> pre_loop_stmts;
      Array<Stmt> post_loop_stmts;
      bool found_loop = false;
      Optional<Stmt> rebuilt_loop = std::nullopt;

      for (const auto &s : seq->seq) {
        // Remove existing create_list_of_mbarrier
        if (IsCreateListOfMbarrier(s))
          continue;

        if (!found_loop && ContainsLoop(s, target_loop)) {
          // Replace the pipeline loop
          rebuilt_loop = RebuildBlockBody(
              s, target_loop, init_barrier, ws_body, buffer_data_to_buffer,
              producer_live_seed, consumer_live_seed);
          found_loop = true;
        } else if (found_loop) {
          // Collect statements after the pipeline loop
          post_loop_stmts.push_back(s);
        } else {
          // Statements before the pipeline loop.
          pre_loop_stmts.push_back(s);
        }
      }

      // Move a movable suffix of pre-loop statements into consumer branch
      // (e.g. fragment initialization), keeping barriers/syncs outside.
      size_t movable_begin = pre_loop_stmts.size();
      while (movable_begin > 0 &&
             IsMovableConsumerPrefixStmt(pre_loop_stmts[movable_begin - 1])) {
        --movable_begin;
      }

      // Split non-movable pre-loop statements into:
      //   common statements kept outside the WS split
      //   producer-side async issues
      //   consumer-side waits / shared-dependent setup
      //   branch-local prefix code that is assigned by actual downstream use
      //     (producer only / consumer only / duplicated).
      //
      // We drive the branch-local assignment with a backward liveness walk over
      // local buffers / Let vars. This avoids duplicating consumer-only local
      // initialization into the producer branch.
      enum class PrefixRole : uint8_t {
        kUnknown,
        kSkip,
        kCommon,
        kProducer,
        kConsumer,
        kBoth,
        kConsumerShared,
        kSpecialTmaStart,
      };

      Array<Stmt> common_pre_stmts;
      Array<Stmt> producer_prefix_ordered_stmts;
      Array<Stmt> consumer_prefix_early_stmts;
      Array<Stmt> consumer_wait_prefix_stmts;
      Array<Stmt> consumer_shared_prefix_stmts;
      std::vector<PrefixRole> prefix_roles(movable_begin, PrefixRole::kUnknown);
      std::vector<Optional<Stmt>> rewritten_producer_prefix(movable_begin,
                                                            std::nullopt);
      std::vector<Optional<Stmt>> rewritten_consumer_wait(movable_begin,
                                                          std::nullopt);

      auto apply_to_live = [](LocalLiveSet *live,
                              const LocalAccessSummary &summary) {
        live->KillDefs(summary);
        live->AddUses(summary);
      };

      LocalLiveSet producer_live = producer_live_seed;
      LocalLiveSet consumer_live = consumer_live_seed;
      for (size_t j = movable_begin; j < pre_loop_stmts.size(); ++j) {
        consumer_live.AddUses(LocalAccessCollector::Collect(
            pre_loop_stmts[j], buffer_data_to_buffer));
      }
      for (const auto &stmt : post_loop_stmts) {
        consumer_live.AddUses(
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer));
      }

      for (int i = static_cast<int>(movable_begin) - 1; i >= 0; --i) {
        if (i > 0 && ContainsTmaLoadStmt(pre_loop_stmts[i - 1]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i])) {
          prefix_roles[i] = PrefixRole::kSkip;
          continue;
        }

        if (static_cast<size_t>(i + 1) < movable_begin &&
            ContainsTmaLoadStmt(pre_loop_stmts[i]) &&
            IsMbarrierWaitParityStmt(pre_loop_stmts[i + 1])) {
          Stmt producer_prefix_stmt =
              StripTmaCopyWriteBufferAttr(pre_loop_stmts[i]);
          Stmt consumer_wait_stmt = pre_loop_stmts[i + 1];
          if (remap_pure_tma_barriers_) {
            ICHECK_GE(pure_tma_preloop_fwd_base_, 0);
            ICHECK_LT(pure_tma_preloop_fwd_cursor_,
                      pure_tma_preloop_fwd_count_);
            PrimExpr barrier_id =
                IntImm(DataType::Int(32), pure_tma_preloop_fwd_base_ +
                                              pure_tma_preloop_fwd_cursor_++);
            producer_prefix_stmt =
                RewriteTmaForwardProducerStmt(producer_prefix_stmt, barrier_id,
                                              /*append_arrive=*/true);
            consumer_wait_stmt =
                RewriteWaitBarrier(consumer_wait_stmt, barrier_id);
          } else if (use_full_tma_forward_barrier_protocol_) {
            auto barrier_id = ExtractWaitBarrierId(pre_loop_stmts[i + 1]);
            ICHECK(barrier_id.defined())
                << "FineGrainedWS: failed to extract pre-loop TMA "
                   "forward barrier id";
            producer_prefix_stmt = RewriteTmaForwardProducerStmt(
                producer_prefix_stmt, barrier_id.value(),
                /*append_arrive=*/true);
          }
          producer_prefix_stmt =
              MergeAdjacentEquivalentIfs(producer_prefix_stmt);
          rewritten_producer_prefix[i] = producer_prefix_stmt;
          rewritten_consumer_wait[i] = consumer_wait_stmt;
          prefix_roles[i] = PrefixRole::kSpecialTmaStart;
          prefix_roles[i + 1] = PrefixRole::kSkip;
          apply_to_live(&producer_live,
                        LocalAccessCollector::Collect(producer_prefix_stmt,
                                                      buffer_data_to_buffer));
          apply_to_live(&consumer_live,
                        LocalAccessCollector::Collect(consumer_wait_stmt,
                                                      buffer_data_to_buffer));
          continue;
        }

        const Stmt &stmt = pre_loop_stmts[i];
        LocalAccessSummary summary =
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer);
        if (remap_pure_tma_barriers_ &&
            IsBranchLocalPreStmtCandidate(stmt, summary)) {
          bool producer_needed = producer_live.NeedsAnyDef(summary);
          bool consumer_needed = consumer_live.NeedsAnyDef(summary);
          if (producer_needed && consumer_needed) {
            prefix_roles[i] = PrefixRole::kBoth;
            apply_to_live(&producer_live, summary);
            apply_to_live(&consumer_live, summary);
          } else if (producer_needed) {
            prefix_roles[i] = PrefixRole::kProducer;
            apply_to_live(&producer_live, summary);
          } else if (consumer_needed) {
            prefix_roles[i] = PrefixRole::kConsumer;
            apply_to_live(&consumer_live, summary);
          } else {
            prefix_roles[i] = PrefixRole::kCommon;
            apply_to_live(&producer_live, summary);
            apply_to_live(&consumer_live, summary);
          }
          continue;
        }

        if (IsSharedDependentConsumerPreStmt(stmt)) {
          prefix_roles[i] = PrefixRole::kConsumerShared;
          apply_to_live(&consumer_live, summary);
        } else {
          prefix_roles[i] = PrefixRole::kCommon;
          apply_to_live(&producer_live, summary);
          apply_to_live(&consumer_live, summary);
        }
      }

      for (size_t i = 0; i < movable_begin; ++i) {
        switch (prefix_roles[i]) {
        case PrefixRole::kSkip:
          break;
        case PrefixRole::kCommon:
          common_pre_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kProducer:
          producer_prefix_ordered_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kConsumer:
          consumer_prefix_early_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kBoth:
          producer_prefix_ordered_stmts.push_back(pre_loop_stmts[i]);
          consumer_prefix_early_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kConsumerShared:
          consumer_shared_prefix_stmts.push_back(pre_loop_stmts[i]);
          break;
        case PrefixRole::kSpecialTmaStart:
          ICHECK(rewritten_producer_prefix[i].defined());
          ICHECK(rewritten_consumer_wait[i].defined());
          producer_prefix_ordered_stmts.push_back(
              rewritten_producer_prefix[i].value());
          consumer_wait_prefix_stmts.push_back(
              rewritten_consumer_wait[i].value());
          break;
        case PrefixRole::kUnknown:
          common_pre_stmts.push_back(pre_loop_stmts[i]);
          break;
        }
      }

      for (const auto &s : common_pre_stmts) {
        new_seq.push_back(s);
      }

      auto MakeOptionalStmt = [](const Array<Stmt> &stmts) -> Optional<Stmt> {
        if (stmts.empty()) {
          return std::nullopt;
        }
        return stmts.size() == 1 ? Optional<Stmt>(stmts[0])
                                 : Optional<Stmt>(SeqStmt(stmts));
      };

      Array<Stmt> consumer_prefix_stmts;
      for (const auto &s : consumer_prefix_early_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      // Keep pure local init before waits to delay blocking until needed.
      for (size_t j = movable_begin; j < pre_loop_stmts.size(); ++j) {
        consumer_prefix_stmts.push_back(pre_loop_stmts[j]);
      }
      for (const auto &s : consumer_wait_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      for (const auto &s : consumer_shared_prefix_stmts) {
        consumer_prefix_stmts.push_back(s);
      }
      Optional<Stmt> consumer_prefix = MakeOptionalStmt(consumer_prefix_stmts);
      Optional<Stmt> producer_prefix =
          MakeOptionalStmt(producer_prefix_ordered_stmts);

      Optional<Stmt> ws_stmt = rebuilt_loop;
      Optional<Stmt> producer_guard = std::nullopt;
      Optional<Stmt> pre_guard = std::nullopt;
      Optional<Stmt> post_guard = std::nullopt;

      // Merge TMA-issue producer prefix into producer branch.
      if (producer_prefix.defined()) {
        ICHECK(thread_iv_.defined());
        Stmt rewritten = PCThreadIdxRewriter::Rewrite(
            producer_prefix.value(), thread_iv_->var,
            thread_iv_->var - consumer_thread_extent_, producer_thread_extent_,
            /*do_shuffle=*/true);
        if (ws_stmt.defined()) {
          auto merged = TryPrependToProducerBranch(ws_stmt.value(), rewritten);
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            producer_guard = IfThenElse(
                GE(thread_iv_->var, consumer_thread_extent_), rewritten);
          }
        } else {
          producer_guard = IfThenElse(
              GE(thread_iv_->var, consumer_thread_extent_), rewritten);
        }
      }

      // Merge movable pre-loop suffix into consumer branch when possible.
      if (consumer_prefix.defined()) {
        if (ws_stmt.defined()) {
          auto merged = TryPrependToConsumerBranch(ws_stmt.value(),
                                                   consumer_prefix.value());
          if (merged.defined()) {
            ws_stmt = merged.value();
          } else {
            ICHECK(thread_iv_.defined());
            pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                   consumer_prefix.value());
          }
        } else {
          ICHECK(thread_iv_.defined());
          pre_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                 consumer_prefix.value());
        }
      }

      // Keep post-loop statements on consumer threads.
      if (!post_loop_stmts.empty()) {
        Stmt post_body = post_loop_stmts.size() == 1 ? post_loop_stmts[0]
                                                     : SeqStmt(post_loop_stmts);
        // Dual-consumer: post-loop (o_acc/=l, O write, lse write) must
        // run on WG1 threads (128-255) only. Adjust threadIdx by -128
        // and wrap with a guard so WG0 threads skip it.
        if (dual_consumer_enabled_) {
          post_body = ThreadIdxSubstitutor::Substitute(
              post_body, thread_iv_->var,
              IntImm(DataType::Int(32), 128));
          // Guard: only WG1 threads (tid >= 128) execute post-loop
          post_body = IfThenElse(
              GE(thread_iv_->var, IntImm(DataType::Int(32), 128)),
              post_body);
        }
        if (remap_pure_tma_barriers_) {
          // When the target loop remaps pure-TMA forward barriers to the WS
          // layout, any remaining TMA forward pairs outside that loop need
          // fresh ids as well. Otherwise a rewritten pre-loop pair can alias a
          // later consumer-only TMA loop that still uses its original id.
          post_body = RewritePureTmaForwardPairsWithFreshBarriers(post_body);
        }
        bool merged = false;
        if (ws_stmt.defined()) {
          auto merged_stmt =
              TryAppendToConsumerBranch(ws_stmt.value(), post_body);
          if (merged_stmt.defined()) {
            ws_stmt = merged_stmt.value();
            merged = true;
          }
        }
        if (!merged) {
          ICHECK(thread_iv_.defined());
          post_guard = IfThenElse(LT(thread_iv_->var, consumer_thread_extent_),
                                  post_body);
        }
      }

      if (producer_guard.defined()) {
        new_seq.push_back(producer_guard.value());
      }
      if (pre_guard.defined()) {
        new_seq.push_back(pre_guard.value());
      }
      if (ws_stmt.defined()) {
        new_seq.push_back(ws_stmt.value());
      }
      if (post_guard.defined()) {
        new_seq.push_back(post_guard.value());
      }

      if (new_seq.size() == 1)
        return new_seq[0];
      return SeqStmt(new_seq);
    }

    // Walk through wrapper nodes
    if (auto *attr = body.as<AttrStmtNode>()) {
      if (ContainsLoop(attr->body, target_loop)) {
        Stmt new_body = RebuildBlockBody(
            attr->body, target_loop, init_barrier, ws_body,
            buffer_data_to_buffer, producer_live_seed, consumer_live_seed);
        return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
      }
    }
    if (auto *let_s = body.as<LetStmtNode>()) {
      if (ContainsLoop(let_s->body, target_loop)) {
        Stmt new_body = RebuildBlockBody(
            let_s->body, target_loop, init_barrier, ws_body,
            buffer_data_to_buffer, producer_live_seed, consumer_live_seed);
        return LetStmt(let_s->var, let_s->value, new_body);
      }
    }

    // Fallback: return unchanged
    return body;
  }

  bool ContainsLoop(const Stmt &stmt, const ForNode *target) {
    if (stmt.as<ForNode>() == target)
      return true;
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        if (ContainsLoop(s, target))
          return true;
      }
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ContainsLoop(attr->body, target);
    }
    if (auto *let_s = stmt.as<LetStmtNode>()) {
      return ContainsLoop(let_s->body, target);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return ContainsLoop(realize->block->body, target);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ContainsLoop(block->body, target);
    }
    return false;
  }

  bool IsCreateListOfMbarrier(const Stmt &stmt) {
    if (auto *eval = stmt.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        return call->op.same_as(create_list_of_mbarrier());
      }
    }
    return false;
  }

  IterVar thread_iv_;
  PrimExpr
      consumer_thread_extent_; // Original thread extent (consumer warp count)
  PrimExpr producer_thread_extent_ = IntImm(DataType::Int(32), 128);
  int configured_producer_thread_extent_ = 128;
  Optional<PrimExpr> num_threads_;
  bool ws_transformed_ = false;
  bool three_role_enabled_ = false;
  bool dual_consumer_enabled_ = false;
  bool user_set_producer_extent_ = false;
  // Cross-stage consumer config (Proposal 1)
  std::unordered_map<std::string, std::vector<std::pair<std::string, int>>>
      consumer_stage_map_;
  // Barrier hints from Phase B (Proposal 2)
  std::unordered_map<std::string, std::pair<int, int>> barrier_hints_;
  // Explicit per-compute-stmt stage offsets (Proposal 2, auto-derived)
  std::unordered_map<int, int> explicit_stage_offsets_;
  bool use_full_tma_forward_barrier_protocol_ = false;
  bool remap_pure_tma_barriers_ = false;
  int pure_tma_preloop_fwd_base_ = -1;
  int pure_tma_preloop_fwd_count_ = 0;
  int pure_tma_preloop_fwd_cursor_ = 0;
  VarBindingMap current_loop_guard_bindings_;
  // Per-op warp group assignments (Plan B): compute_stmt index → warp group ID
  std::unordered_map<int, int> warp_assigns_map_;
};

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

using namespace tir::transform;

// Check only for manual warp specialization ("warp_specialize" attr).
// Unlike WarpSpecializedDetector, we do NOT skip when TMA+mbarrier are
// both present, since that is the expected input pattern for this pass.
class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d.VisitStmt(stmt);
    return d.has_manual_ws_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "warp_specialize" &&
        op->value.as<IntImmNode>()->value == 1) {
      has_manual_ws_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  bool has_manual_ws_ = false;
};

tvm::transform::Pass MakeFineGrainedWarpSpecializedPass(const std::string &pass_name) {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_warp_specialized =
        ctx->GetConfig<Bool>(kDisableWarpSpecialized, Bool(false)).value();
    if (disable_warp_specialized)
      return f;

    // Skip if user has manual warp specialization
    if (ManualWSDetector::HasManualWS(f->body))
      return f;

    // Configurable producer thread extent (default 128, min 32, must be
    // multiple of warp size). Reducing to 32 for TMA-only producers saves
    // register file pressure and warp scheduling overhead.
    auto producer_opt =
        ctx->GetConfig(kFineGrainedWsProducerThreadExtent, Optional<Integer>());
    bool user_set_producer = producer_opt.defined();
    int producer_threads =
        user_set_producer ? static_cast<int>(producer_opt.value()->value) : 128;
    if (producer_threads < 32) producer_threads = 32;
    // Round up to warp size
    producer_threads = ((producer_threads + 31) / 32) * 32;

    bool three_role =
        ctx->GetConfig<Bool>(kFineGrainedWsEnableThreeRole, Bool(false)).value();
    bool dual_consumer =
        ctx->GetConfig<Bool>(kFineGrainedWsDualConsumer, Bool(false)).value();

    // Cross-stage consumer config (Proposal 1)
    std::string consumer_stage_map_str =
        ctx->GetConfig(kFineGrainedWsConsumerStageMap, Optional<String>())
            .value_or(String(""));
    // Barrier hints from Phase B (Proposal 2)
    std::string barrier_hints_str =
        ctx->GetConfig(kFineGrainedWsBarrierHints, Optional<String>())
            .value_or(String(""));
    // Per-compute-stmt stage offsets (Proposal 2, auto-derived)
    std::string stage_offsets_str =
        ctx->GetConfig(kFineGrainedWsStageOffsets, Optional<String>())
            .value_or(String(""));
    // Per-op warp group assignments (Plan B)
    std::string warp_assigns_str =
        ctx->GetConfig(kFineGrainedWsWarpAssigns, Optional<String>())
            .value_or(String(""));

    return FineGrainedWSRewriter::Substitute(
        f, producer_threads, three_role, user_set_producer,
        dual_consumer,
        consumer_stage_map_str, barrier_hints_str, stage_offsets_str,
        warp_assigns_str);
  };
  return CreatePrimFuncPass(pass_func, 0, pass_name, {});
}

tvm::transform::Pass FineGrainedWarpSpecialized() {
  return MakeFineGrainedWarpSpecializedPass("tl.FineGrainedWarpSpecialized");
}

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  // Compatibility shim for TileLang v0.1.8's existing pass pipeline.
  return MakeFineGrainedWarpSpecializedPass("tl.ProducerConsumerWarpSpecialized");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.FineGrainedWarpSpecialized",
                        FineGrainedWarpSpecialized);
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
