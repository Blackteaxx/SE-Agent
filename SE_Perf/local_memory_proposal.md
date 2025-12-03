# SE-Agent Local Memory Manager 设计文档

[cite_start]本文档详细描述了 SE-Agent 在进化算法迭代过程中维护 **Local Memory (局部记忆)** 的设计方案。该模块旨在通过维护短期工作记忆，解决“盲目试错”与“经验遗忘”问题，提升算法的搜索效率 [cite: 7, 19]。

-----

## 1\. 核心目标 (Core Objectives)

Local Memory 充当 Agent 的“短期工作记忆”，主要承担以下三项核心功能：

1.  **全局状态感知 (Global Awareness)**：
    记录当前的进化代数、最佳性能指标及对应的解 ID，使模型保持对当前优化进度的感知。
2.  **方向引导 (Direction Guidance)**：
    [cite_start]实时维护“禁忌表（Tabu List）”与“推荐列表”。通过记录已尝试过的高层策略及其成败结果，防止 Agent 在同一个错误方向上反复试错，或忽略已验证的高潜方向 [cite: 24, 151]。
3.  **经验沉淀 (Reasoning Bank)**：
    [cite_start]基于 **ReasoningBank** 的思想，从具体的代码轨迹中提炼可迁移的成功策略或失败教训 [cite: 8, 154][cite_start]。不同于原始轨迹，这里存储的是经过蒸馏的、带有代码证据（Evidence）的结构化知识 [cite: 153-155]。

-----

## 2\. 数据结构设计 (Memory Schema)

该结构由 Python 后台作为 **Backend Storage** 进行全量维护。在实际发送给 LLM 时，将根据此结构进行渲染。

```json
{
  "global_status": {
    "current_generation": 5,           // 当前进化代数
    "best_runtime": "120ms",           // 当前最佳运行时间
    "best_solution_id": "Gen3_Sol_4",  // 最佳解的来源ID
    "current_approach": "Dynamic Programming with Bitmask" // 当前算法流派简述
  },

  // [策略层] 实时维护的“方向标”
  // 更新逻辑：每次 Extraction 时，模型读取旧列表，根据新结果进行修改、删除或新增
  "attempted_directions": [
    {
      "direction": "Use distinct IDs for visited nodes",
      "outcome": "Failed",
      "source_ref": "Gen_2_Sol_1",
      "evidence": "TLE on Test Case 15. Runtime degraded to 2000ms+."
    },
    {
      "direction": "Switch standard I/O to fast I/O",
      "outcome": "Success",
      "source_ref": "Gen_4_Sol_3",
      "evidence": "Consistent 10ms gain. Verified stable."
    }
  ],

  // [知识层] 经验库 (Reasoning Bank)
  [cite_start]// 对应 ReasoningBank 的核心结构：Title, Description, Content [cite: 156-158]
  // 更新逻辑：默认 Append，超长时触发 Compress
  "reasoning_bank": [
    {
      "type": "Success", 
      "title": "Bitwise Operation Optimization",
      "description": "Replace modulo operator with bitwise AND for powers of 2.",
      "content": "Using `x & (MOD - 1)` instead of `x % MOD` reduced constant factor.",
      "related_operator": "Refinement",
      "source_ref": {
        "generation": 3,
        "solution_id": "Sol_5",
        "parent_id": "Gen_2_Sol_2"
      },
      "evidence": {
        "code_change": "Changed `dp[i] % 1024` -> `dp[i] & 1023` at line 45.", 
        "metrics_delta": "Runtime: 150ms -> 120ms (-20%)",
        "context": "Effective when MOD=1024 (power of 2)."
      }
    }
    // ... 更多 Memory Items
  ]
}
```

-----

## 3\. 运行逻辑与生命周期 (Lifecycle)

[cite_start]整个过程是一个闭环：Agent 利用记忆指导行动，随后分析新产生的经验，并将其更新回记忆库 [cite: 81-82]。

### 阶段 I：渲染与使用 (Usage)

[cite_start]本阶段遵循“重后台存储，轻前台展示”的原则，将 JSON 数据转换为自然语言 Prompt，即 ReasoningBank 中的“Memory Retrieval”步骤 [cite: 168]。

  * **触发时机**：在 Mutation/Crossover 算子准备生成新代码之前。
  * **执行逻辑**：读取 Python 中的 JSON 对象，将其扁平化渲染为 Markdown 格式（去除冗余 Key，保留核心 Value），并注入到 System Prompt 中。

### 阶段 II：提炼与更新 (Extraction & Update)

[cite_start]这是系统的核心环节，负责处理“噪声过滤”与“双层更新”。该机制借鉴了 ReasoningBank 中从轨迹中提取 Success/Failure 信号的方法，将原始轨迹转化为结构化记忆 [cite: 170-172]。

  * **触发时机**：当新解 $S_{new}$ 完成轨迹总结（Trajectory Summary）后。

  * **输入数据**：

      * Problem Description
      * Old Code vs New Code (Diff)
      * Old Metrics vs New Metrics (e.g., Time: 150ms -\> 148ms)
      * Current Directions (旧的方向列表，用于增量更新)

  * **预处理逻辑 (Python 端)**：
    [cite_start]利用指标进行初步过滤，类似 LLM-as-a-Judge 的二分类判定 [cite: 171]。

    1.  **Metric Analysis**：计算性能差异 `perf_diff = perf_old - perf_new`。
    2.  **Extraction Branch (分支判定)**：
          * [cite_start]**Success Branch (`perf_diff > 0`)**：进入成功经验提取流程。模型需进一步判断该提升是算法优化的有效结果，还是环境波动的噪声 [cite: 172]。
          * [cite_start]**Failure Branch (`perf_diff <= 0`)**：进入失败经验提取流程。模型需判断是真正的策略失败，还是无效的微小波动 [cite: 172]。

  * **LLM Extraction Prompt 设计**：
    模型需同时完成更新 Directions 和提取 Memory Item 两个任务。

<!-- end list -->

```markdown
You are the Memory Manager for an evolutionary coding agent.

## Context
- Old Runtime: 150ms
- New Runtime: 120ms (Improved by 20%) [Outcome: SUCCESS]

## Inputs
1. **Code Diff**: [Diff Content...]
2. **Current Directions**: [JSON List of Attempted Directions...]

## Task
1. **Update Directions**: 
   - If this approach was already in "Current Directions", update its evidence and status.
   - If this is a new approach, add it.
   - Remove obsolete or duplicate directions.
   - Keep the list concise and high-level.

2. **Create Reasoning Item**:
   - Extract the specific insight (Success logic or Failure lesson).
   - Provide semantic evidence (Not just raw diff, but what changed logic-wise).

## Output JSON
{
  "updated_directions": [ ... ],
  "new_memory_item": { ... } // Or null if insight is trivial
}
```

### 阶段 III：维护与压缩 (Maintenance)

[cite_start]Directions、ReasoningBank 和 Best Solution 采用独立的维护策略，类似于 ReasoningBank 中的 Memory Consolidation [cite: 174]。

  * **Directions 维护 (全量替换)**：
      * 直接使用 LLM 输出的 `updated_directions` **完全替换** 后台存储中的旧列表。这保证了 Directions 始终是最新的、经过整理去重的状态。
  * **Reasoning Bank 维护 (增量 + 压缩)**：
      * **Append**：将 `new_memory_item` 追加到列表末尾。
      * **Token Check**：每次追加后，检查总 Memory Token 是否超过阈值（例如 1500 tokens）。
      * **Compress (触发式)**：如果超过阈值，调用 `summarize_memory()` 进行压缩：
          * **保留**：Top-3 Success (按提升幅度排序) + Top-2 Failure (按严重程度)。
          * **丢弃**：过时的、提升微不足道的条目。
          * **归纳**：将相似的条目合并（例如：将 3 条关于 I/O 优化的零散记录合并为 1 条通用策略）。

-----

## 4\. 实现流程图 (Pipeline)

整个流程形成一个闭环，确保经验不断积累且不爆炸。

1.  **Code Eval**：
    运行评测系统，获取 `Runtime`, `Memory`, `Status` (AC/TLE/RE)。
2.  **Metric Branch**：
    Python 端根据指标变化计算 `perf_diff`，决定进入 Success 分支还是 Failure 分支。
3.  **LLM Extraction**：
    构造 Prompt，发送 `System Prompt` + `Diff` + `Current Directions` 给 LLM。
4.  **Update State**：
      * `local_memory['attempted_directions'] = llm_response['updated_directions']` (全量更新)
      * `if llm_response['new_memory_item']: local_memory['reasoning_bank'].append(...)` (增量追加)
5.  **Auto-Compress**：
      * `if count_tokens(local_memory) > LIMIT: compress(local_memory)` (动态维护)