【约束】
这里是给你的限制
Heddle作为tilelang的插件执行。但因为环境限制，无法安装tilelang。所以Heddle也无法运行验证
你所做的代码修改若可执行，应进行最小单元（文件、函数级别）的测试验证保证可行。跳过Heddle的整体验证
改代码前，请先阅读tilelang Heddle内的相关代码

【说明】
/Users/xiaoying/Desktop/aicode/Heddle/Heddle/heddle/sched 
  uler/cp_sat.py 这个文件中 UnifiedScheduler的solve函数用于 
  做SMT求解消费者排序。但是目前，缺少warp分配的相关建模，以 
  及现有消费者排序逻辑存在问题（barrier问题），并且无法替换 
  forOp的body。
【任务】
帮我修复下已知问题，并完成 warp_specialized 的SMT约束添加。