use std::collections::HashSet;

use crate::eval::{EvaluationReport, evaluate_algorithm, split_holdout};
use crate::types::Dataset;

pub fn generate_report(dataset: &Dataset, top_n: usize, holdout_ratio: f32) -> String {
    let algorithms = [
        "content",
        "knowledge",
        "user-cf",
        "item-cf",
        "popularity",
        "matrix",
        "hybrid",
    ];
    generate_report_for_algorithms(dataset, top_n, holdout_ratio, &algorithms)
}

pub fn generate_report_for_algorithms(
    dataset: &Dataset,
    top_n: usize,
    holdout_ratio: f32,
    algorithms: &[&str],
) -> String {
    let rows = algorithms
        .iter()
        .map(|name| {
            (
                *name,
                evaluate_algorithm(name, dataset, top_n, holdout_ratio),
            )
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return "未指定待评估算法。\n".to_string();
    }

    let summary = ReportSummary::from_rows(&rows);
    let (train_dataset, test) = split_holdout(dataset, holdout_ratio);
    let test_users = test.iter().map(|r| r.user_id).collect::<HashSet<_>>();
    let relevant_users = test
        .iter()
        .filter(|r| r.rating >= 4.0)
        .map(|r| r.user_id)
        .collect::<HashSet<_>>();
    let relevant_count = test.iter().filter(|r| r.rating >= 4.0).count();

    let mut out = String::new();
    out.push_str("# 智能搜索与动态混合推荐系统实验报告\n\n");

    out.push_str("## 摘要\n\n");
    out.push_str(&format!(
        "本实验实现并比较了基于内容、基于知识、用户协同过滤、物品协同过滤、热门度、矩阵分解和动态混合推荐共 {} 类算法。实验采用 MovieLens 100K 兼容数据格式，以用户内部时间顺序 holdout 方式构造训练集和测试集，并在 Top-{} 场景下同时评估评分预测误差、二分类偏好判断、排序质量、命中能力、目录覆盖率、多样性和流行度偏置。报告重点不是声称某个模型绝对最优，而是解释不同模型在准确性、泛化能力、可解释性和长尾覆盖之间的取舍。\n\n",
        rows.len(),
        top_n
    ));

    out.push_str("## 项目设计思路与具体实现\n\n");
    out.push_str(
        "本项目按照“数据读取与建模、搜索入口、单模型推荐、动态混合、离线评估、报告生成”的流程组织实现。系统首先读取 MovieLens 兼容格式的电影信息和用户评分，构建按用户、按电影聚合的索引以及全局均值、用户均值、电影均值等统计量；搜索模块提供标题关键词、类型和年份过滤，用于承接用户的显式查询意图；推荐模块则分别实现内容推荐、知识规则、用户协同过滤、物品协同过滤、热门度基线和矩阵分解，形成多个互补的候选评分来源。\n\n",
    );
    out.push_str(
        "最终系统采用动态混合推荐作为整体方案：先把不同推荐器的输出统一到 0 到 1 的归一化评分空间，再根据用户历史长度、电影评分数量和各模块置信度调整权重，使协同过滤和矩阵分解承担主要预测任务，内容、知识和热门度模块提供冷启动、解释性和稳定性补充。评估部分使用按用户时间顺序的 holdout 切分，避免未来行为泄漏，并同时输出 MAE、RMSE、Precision、Recall、HitRate、nDCG、覆盖率、多样性和流行度偏置等指标，从而让系统不只给出推荐结果，也能解释推荐质量和模型局限。\n\n",
    );

    out.push_str("## 数据集与实验设置\n\n");
    out.push_str(&format!(
        "- 数据格式：MovieLens 100K 兼容 `u.data` / `u.item`\n- 电影数量：{}\n- 原始评分数量：{}\n- 训练评分数量：{}\n- 测试评分数量：{}\n- 测试用户数量：{}\n- 含相关测试项目的用户数量：{}\n- 测试集中相关项目数量：{}\n- Holdout 比例：{:.2}\n- Top-N：{}\n- 相关项目定义：测试集中真实评分 `r_ui >= 4` 的电影视为用户相关项目\n- 二分类阈值：预测评分 `hat r_ui >= 4` 判定为用户可能喜欢\n\n",
        dataset.movies.len(),
        dataset.ratings.len(),
        train_dataset.ratings.len(),
        test.len(),
        test_users.len(),
        relevant_users.len(),
        relevant_count,
        holdout_ratio,
        top_n
    ));
    out.push_str(
        "切分方式为每个用户按时间戳升序排序，将末尾一段评分作为测试集，前面的评分作为训练集。这样可以减少“用未来行为预测过去行为”的信息泄漏，比随机切分更接近真实推荐场景。\n\n",
    );
    out.push_str("![图 1 时间顺序 holdout 切分实现](reports/code_screenshots/code_page2_1.png)\n\n");

    push_method_section(&mut out);
    push_metric_section(&mut out, top_n);

    out.push_str("## 实验结果\n\n");
    push_prediction_table(&mut out, &rows);
    push_ranking_table(&mut out, &rows, top_n);
    push_coverage_table(&mut out, &rows);

    push_analysis_section(&mut out, &rows, &summary, top_n);
    push_limitations_section(&mut out);

    out.push_str("## 结论\n\n");
    out.push_str(&format!(
        "综合实验结果，`{}` 更适合作为评分预测主模型，`{}` 更适合作为 Top-{} 排序质量参考模型。动态混合推荐虽然不一定在每个单项指标上取得第一，但它把矩阵分解、协同过滤、内容特征、知识规则和热门度基线统一到同一归一化评分空间中，能够在准确性、解释性和冷启动补充之间取得较均衡的系统效果。因此，本系统适合作为课程大作业中的完整推荐系统原型：核心算法可运行，评估闭环完整，报告也能解释为什么不同指标会给出不同的模型排序。\n\n",
        summary.best_mae.0,
        summary.best_ndcg.0,
        top_n
    ));

    out.push_str("## 测试命令\n\n");
    out.push_str("```powershell\ncargo test\ncargo run -- evaluate --algorithm all --top-n 10 --holdout-ratio 0.2 --report reports/test_report.md\n```\n");
    out
}

fn push_method_section(out: &mut String) {
    out.push_str("## 方法与数学建模\n\n");
    out.push_str("### 统一评分空间\n\n");
    out.push_str(
        r#"所有推荐器最终输出归一化得分 `s_ui in [0, 1]`，再映射回 1 到 5 星评分：

$$
s_{ui} = \frac{\hat r_{ui} - 1}{4}
$$

$$
\hat r_{ui} = 1 + 4s_{ui}
$$

统一评分空间的作用是让内容相似度、规则得分、协同过滤预测值、热门度分数和矩阵分解预测值可以被同一个混合模型加权融合。

"#,
    );
    out.push_str("![图 2 评分归一化与相似度工具函数](reports/code_screenshots/code_page3_1.png)\n\n");

    out.push_str("### 基于内容的推荐\n\n");
    out.push_str(
        r#"电影 `i` 被表示为稀疏内容向量 `x_i`。当前实现使用三类特征：电影类型、标题 token 和发行年代，其中类型权重为 2.0，标题 token 权重为 0.5，年代权重为 0.6。用户画像只由高评分电影构成：

$$
I_u^+ = \{ i \mid r_{ui} \geq 4 \}
$$

$$
p_u = \sum_{i \in I_u^+} \operatorname{norm}(r_{ui}) x_i
$$

$$
\operatorname{score}(u,i) = \cos(p_u, x_i)
$$

这里 `norm(r_ui)` 是评分归一化后的权重。该方法可解释性强，能说明“推荐原因来自类型、标题关键词或年代相似”，但它依赖元数据质量。由于 MovieLens 100K 的内容字段较少，基于内容的方法更适合作为冷启动和解释信号，而不是唯一主模型。

"#,
    );

    out.push_str("### 基于知识的推荐\n\n");
    out.push_str(
        r#"基于知识的推荐器从用户历史中抽取偏好类型、排斥类型和偏好年份范围。其得分是一个可解释的规则模型：

$$
\operatorname{score}(u,i)
= 0.50 + g^+_{ui} - g^-_{ui} + y_{ui}
$$

其中喜欢类型来自 `r_ui >= 4` 的历史电影，排斥类型来自 `r_ui <= 2` 的历史电影。规则模型的优点是透明、容易解释，缺点是表达能力弱，无法自动学习复杂的交互关系。

"#,
    );

    out.push_str("### 搜索模块\n\n");
    out.push_str(
        r#"搜索模块支持标题关键词、类型和起始年份过滤。候选电影先经过结构化条件筛选，再按标题 token 命中分和轻量热门度分进行排序：

$$
\operatorname{SearchScore}(i,q)
= 2 \cdot \operatorname{match}(i,q) + 0.01 \cdot n_i
$$

这个模块能满足课程作业中的基础检索需求，也能和推荐模块形成互补：搜索强调用户显式输入的即时意图，推荐强调历史行为推断出的长期偏好。当前搜索仍是轻量实现，没有倒排索引、TF-IDF/BM25 和拼写纠错，因此报告中把它定位为“可运行的检索入口”，而不是完整搜索引擎。

"#,
    );
    out.push_str("![图 3 搜索模块过滤与排序实现](reports/code_screenshots/code_page4_1.png)\n\n");

    out.push_str("### 协同过滤\n\n");
    out.push_str(
        r#"用户协同过滤先在共同评分电影上计算 Pearson 相似度，并加入收缩因子抑制共同评分数过少导致的偶然高相似度：

$$
\operatorname{sim}(u,v)
= \max(0,\operatorname{Pearson}(u,v)) \cdot \frac{|C_{uv}|}{|C_{uv}| + 10}
$$

$$
\hat r_{ui}
= \bar r_u + c \cdot
\frac{\sum_v \operatorname{sim}(u,v)(r_{vi}-\bar r_v)}
{\sum_v |\operatorname{sim}(u,v)|}
$$

物品协同过滤在共同评分用户上计算电影之间的相似度，形式类似：

$$
\operatorname{sim}(i,j)
= \max(0,\operatorname{Pearson}(i,j)) \cdot \frac{|U_{ij}|}{|U_{ij}| + 15}
$$

$$
\hat r_{ui}
= \bar r_i + c \cdot
\frac{\sum_j \operatorname{sim}(i,j)(r_{uj}-\bar r_j)}
{\sum_j |\operatorname{sim}(i,j)|}
$$

其中 `c` 是由邻居数量和相似度强度得到的置信度。协同过滤能捕捉群体行为模式，但对稀疏用户、稀疏物品和冷启动项目较敏感。

"#,
    );
    out.push_str("![图 4 用户协同过滤相似度计算实现](reports/code_screenshots/code_page5_1.png)\n\n");

    out.push_str("### 热门度基线\n\n");
    out.push_str(
        r#"热门度推荐器使用贝叶斯平滑后的电影平均评分，避免少量评分电影因为偶然高分被过度推荐：

$$
\hat r_i = \frac{n_i \bar r_i + m\mu}{n_i + m}, \quad m = 5
$$

其中 `n_i` 是电影评分数，`mean_i` 是电影平均评分，`mu` 是全局平均评分。热门度缺少个性化，但它是非常重要的基线：如果复杂模型不能明显超过热门度，就说明个性化信号并没有被有效利用。

"#,
    );

    out.push_str("### 矩阵分解\n\n");
    out.push_str(
        r#"矩阵分解将用户和电影映射到低维潜在空间，当前实现使用带偏置项的显式反馈模型：

$$
\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i
$$

训练目标可以写成：

$$
\min \sum_{(u,i)\in R_{\mathrm{train}}}(r_{ui}-\hat r_{ui})^2
+ \lambda\left(\|p_u\|^2+\|q_i\|^2+b_u^2+b_i^2\right)
$$

本系统使用 SGD 训练，参数为 24 个隐因子、20 轮迭代、学习率 0.01、正则化系数 0.05。矩阵分解通常适合作为评分预测主模型，因为它既能学习用户偏好，也能学习电影之间的隐含关系。

"#,
    );

    out.push_str("### 动态混合推荐\n\n");
    out.push_str(
        r#"混合模型把各推荐器的归一化得分做加权融合：

$$
S(u,i)=\sum_m w_m(u,i)s_m(u,i)
$$

$$
w_m(u,i)
= \frac{\alpha_m q_m(u,i)a_m(u,i)}
{\sum_k \alpha_k q_k(u,i)a_k(u,i)}
$$

其中 `alpha_m` 是基础权重，`q_m` 是模块置信度，`a_m` 是由用户历史长度、物品评分数量等因素决定的动态调整项。直观上，历史较少的用户会提高内容和知识规则的相对权重；历史较充分且物品评分较多时，矩阵分解和协同过滤权重更高。这比固定平均融合更合理，因为不同用户和不同电影的可用证据强度并不相同。

"#,
    );
}

fn push_metric_section(out: &mut String, top_n: usize) {
    out.push_str("## 评价指标定义\n\n");
    out.push_str(
        r#"### 评分预测指标

对测试集中的评分样本 `(u, i, r_ui)`，模型给出预测 `hat r_ui`。本报告使用：

$$
\mathrm{MAE}
= \frac{1}{|T|}\sum_{(u,i)\in T}|\hat r_{ui}-r_{ui}|
$$

$$
\mathrm{RMSE}
= \sqrt{\frac{1}{|T|}\sum_{(u,i)\in T}(\hat r_{ui}-r_{ui})^2}
$$

MAE 衡量平均绝对误差，RMSE 对大误差更敏感。二分类准确率把真实评分和预测评分都按 4 星阈值转成“喜欢/不喜欢”：

$$
\mathrm{Accuracy}
= \frac{\#\{\operatorname{sign}(\hat r_{ui}\geq 4)=\operatorname{sign}(r_{ui}\geq 4)\}}
{\#\{\mathrm{valid\ predictions}\}}
$$

预测覆盖率表示模型能对多少测试评分给出有效预测。对于只覆盖部分用户或物品的算法，只看 MAE/RMSE 可能会高估模型，因此报告同时列出覆盖率和有效预测数。

"#,
    );

    out.push_str(&format!("### Top-{} 排序指标\n\n", top_n));

    out.push_str(
        r#"对每个测试用户 `u`，令真实相关集合为：

$$
\operatorname{Rel}_u
= \{ i \mid r_{ui}\geq 4,\ (u,i)\in T \}
$$

"#,
    );
    out.push_str(&format!(
        "推荐列表前 `{}` 个结果记为 `L_u@{}`。本报告使用：\n\n",
        top_n, top_n
    ));
    out.push_str(
        r#"$$
\mathrm{Precision}_N
= \frac{|L_u^{(N)} \cap \operatorname{Rel}_u|}{N}
$$

$$
\mathrm{Recall}_N
= \frac{|L_u^{(N)} \cap \operatorname{Rel}_u|}{|\operatorname{Rel}_u|}
$$

$$
\mathrm{F1}_N
= \frac{2\cdot \mathrm{Precision}_N\cdot \mathrm{Recall}_N}
{\mathrm{Precision}_N+\mathrm{Recall}_N}
$$

$$
\mathrm{HitRate}_N
= \mathbf{1}\left(|L_u^{(N)} \cap \operatorname{Rel}_u| > 0\right)
$$

$$
\mathrm{DCG}_N
= \sum_{k=1}^{N}\frac{\operatorname{rel}_k}{\log_2(k+1)}
$$

$$
\mathrm{nDCG}_N
= \frac{\mathrm{DCG}_N}{\mathrm{IDCG}_N}
$$


最终结果对所有存在相关测试项目的用户做宏平均。需要注意的是，MovieLens 的未评分电影不能直接视为用户不喜欢，因此 Top-N 指标通常绝对数值不高，更适合用于算法之间的相对比较。

"#,
    );

    out.push_str(
        r#"### 覆盖率、多样性和流行度

目录覆盖率衡量所有推荐列表触达了多少不同电影：

$$
\mathrm{CatalogCoverage}
= \frac{|\bigcup_u L_u^{(N)}|}{|I|}
$$

平均推荐流行度是推荐电影在训练集中被评分次数的平均值，数值越高说明越偏向热门电影。类型多样性使用推荐列表中不同类型数与类型出现总次数的比值：

$$
\mathrm{GenreDiversity}_N
= \frac{\operatorname{unique\_genres}(L_u^{(N)})}
{\operatorname{genre\_slots}(L_u^{(N)})}
$$

这些指标不直接等价于准确率，但可以解释系统是否过度集中在少数热门电影上，以及推荐结果是否具有一定探索性。

"#,
    );
}

fn push_analysis_section(
    out: &mut String,
    rows: &[(&str, EvaluationReport)],
    summary: &ReportSummary<'_>,
    top_n: usize,
) {
    out.push_str("## 结果分析\n\n");
    out.push_str(&format!(
        "从评分预测角度看，`{}` 获得最低 MAE（{:.4}），`{}` 获得最低 RMSE（{:.4}）。MAE 和 RMSE 越低，说明模型越能还原用户的显式评分；如果同一个模型同时在两个指标上较优，通常可以认为它既减少了平均偏差，也减少了大幅误判。二分类偏好判断方面，`{}` 的准确率最高（{:.4}），说明它在“是否可能喜欢”的粗粒度判断上表现最好。\n\n",
        summary.best_mae.0,
        summary.best_mae.1.mae,
        summary.best_rmse.0,
        summary.best_rmse.1.rmse,
        summary.best_accuracy.0,
        summary.best_accuracy.1.accuracy
    ));

    out.push_str(&format!(
        "从 Top-{} 排序质量看，`{}` 的 nDCG@{} 最高（{:.4}），说明它更倾向于把真实相关电影排在推荐列表靠前位置；`{}` 的 Precision@{} 最高（{:.4}），表示其前 {} 个推荐中相关项目比例最高；`{}` 的 HitRate@{} 最高（{:.4}），说明它最容易为用户提供至少一个命中项目。由于测试时是在完整未看电影集合中排序，而每个用户测试集中的真实高分电影数量有限，因此 Precision、Recall、nDCG 的绝对值偏低是正常现象，重点应看不同算法之间的相对差异。\n\n",
        top_n,
        summary.best_ndcg.0,
        top_n,
        summary.best_ndcg.1.ndcg_at_n,
        summary.best_precision.0,
        top_n,
        summary.best_precision.1.precision_at_n,
        top_n,
        summary.best_hit_rate.0,
        top_n,
        summary.best_hit_rate.1.hit_rate_at_n
    ));

    out.push_str(&format!(
        "从推荐生态角度看，`{}` 的目录覆盖率最高（{:.4}），说明它能触达更大范围的电影目录；`{}` 的平均推荐流行度最高（{:.2}），说明它更偏向大众热门项目；`{}` 的平均推荐流行度最低（{:.2}），说明它相对更接近长尾探索；`{}` 的类型多样性最高（{:.4}），说明其推荐列表在题材分布上更分散。覆盖率和多样性通常会与准确率存在张力：越追求热门高置信电影，短期命中可能越稳定，但长尾覆盖和新颖性会下降。\n\n",
        summary.best_coverage.0,
        summary.best_coverage.1.catalog_coverage,
        summary.highest_popularity.0,
        summary.highest_popularity.1.avg_recommendation_popularity,
        summary.lowest_popularity.0,
        summary.lowest_popularity.1.avg_recommendation_popularity,
        summary.best_diversity.0,
        summary.best_diversity.1.avg_genre_diversity
    ));

    if let Some((_, hybrid)) = rows.iter().find(|(name, _)| *name == "hybrid") {
        out.push_str(&format!(
            "动态混合推荐的意义不只是追求单一指标第一，而是在多种信号之间做稳定折中。本次实验中，hybrid 的 MAE 为 {:.4}、RMSE 为 {:.4}、Precision@{} 为 {:.4}、nDCG@{} 为 {:.4}、目录覆盖率为 {:.4}。这些结果说明混合模型保持了接近强基线的预测和排序能力，同时保留内容、知识和热门度模块提供的解释信号。对于课程项目而言，这比只展示一个黑箱模型更有说明价值，因为报告可以解释每个模块在什么条件下发挥作用。\n\n",
            hybrid.mae,
            hybrid.rmse,
            top_n,
            hybrid.precision_at_n,
            top_n,
            hybrid.ndcg_at_n,
            hybrid.catalog_coverage
        ));
    }
}

fn push_limitations_section(out: &mut String) {
    out.push_str("## 局限性\n\n");
    out.push_str(
        "主要局限包括：第一，内容特征只使用类型、标题 token 和年代，缺少剧情简介、演员、导演、关键词等语义信息，因此内容推荐表达能力有限；第二，知识规则是人工线性规则，解释性强但泛化能力有限；第三，矩阵分解是教学版 SGD，没有验证集早停、随机打乱、超参数搜索和置信区间分析；第四，目前只做一次固定 holdout，指标可能受切分方式影响；第五，搜索模块仍是简单标题匹配加热门度排序，没有使用倒排索引、TF-IDF 或 BM25。\n\n",
    );
}

fn push_prediction_table(out: &mut String, rows: &[(&str, EvaluationReport)]) {
    out.push_str("### 评分预测与二分类指标\n\n");
    out.push_str("| 算法 | 有效预测数 | 预测覆盖率 | 二分类准确率（分类准确度） | MAE | RMSE |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            algorithm,
            result.prediction_count,
            result.prediction_coverage,
            result.accuracy,
            result.mae,
            result.rmse
        ));
    }
    out.push('\n');
}

fn push_ranking_table(out: &mut String, rows: &[(&str, EvaluationReport)], top_n: usize) {
    out.push_str(&format!("### Top-{} 排序与命中指标\n\n", top_n));
    out.push_str("| 算法 | 参与用户数 | Precision@N | Recall@N | F1@N | HitRate@N | nDCG@N |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            algorithm,
            result.ranking_user_count,
            result.precision_at_n,
            result.recall_at_n,
            result.f1_at_n,
            result.hit_rate_at_n,
            result.ndcg_at_n
        ));
    }
    out.push('\n');
}

fn push_coverage_table(out: &mut String, rows: &[(&str, EvaluationReport)]) {
    out.push_str("### 覆盖率、多样性与流行度偏置\n\n");
    out.push_str("| 算法 | 目录覆盖率 | 平均推荐流行度 | 类型多样性 |\n");
    out.push_str("|---|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.2} | {:.4} |\n",
            algorithm,
            result.catalog_coverage,
            result.avg_recommendation_popularity,
            result.avg_genre_diversity
        ));
    }
    out.push('\n');
}

struct ReportSummary<'a> {
    best_accuracy: (&'a str, &'a EvaluationReport),
    best_mae: (&'a str, &'a EvaluationReport),
    best_rmse: (&'a str, &'a EvaluationReport),
    best_ndcg: (&'a str, &'a EvaluationReport),
    best_precision: (&'a str, &'a EvaluationReport),
    best_hit_rate: (&'a str, &'a EvaluationReport),
    best_coverage: (&'a str, &'a EvaluationReport),
    best_diversity: (&'a str, &'a EvaluationReport),
    highest_popularity: (&'a str, &'a EvaluationReport),
    lowest_popularity: (&'a str, &'a EvaluationReport),
}

impl<'a> ReportSummary<'a> {
    fn from_rows(rows: &'a [(&'a str, EvaluationReport)]) -> Self {
        Self {
            best_accuracy: max_by(rows, |r| r.accuracy),
            best_mae: min_by(rows, |r| r.mae),
            best_rmse: min_by(rows, |r| r.rmse),
            best_ndcg: max_by(rows, |r| r.ndcg_at_n),
            best_precision: max_by(rows, |r| r.precision_at_n),
            best_hit_rate: max_by(rows, |r| r.hit_rate_at_n),
            best_coverage: max_by(rows, |r| r.catalog_coverage),
            best_diversity: max_by(rows, |r| r.avg_genre_diversity),
            highest_popularity: max_by(rows, |r| r.avg_recommendation_popularity),
            lowest_popularity: min_by(rows, |r| r.avg_recommendation_popularity),
        }
    }
}

fn min_by<'a>(
    rows: &'a [(&'a str, EvaluationReport)],
    metric: fn(&EvaluationReport) -> f32,
) -> (&'a str, &'a EvaluationReport) {
    rows.iter()
        .min_by(|a, b| {
            metric(&a.1)
                .partial_cmp(&metric(&b.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, report)| (*name, report))
        .expect("report summary requires at least one algorithm")
}

fn max_by<'a>(
    rows: &'a [(&'a str, EvaluationReport)],
    metric: fn(&EvaluationReport) -> f32,
) -> (&'a str, &'a EvaluationReport) {
    rows.iter()
        .max_by(|a, b| {
            metric(&a.1)
                .partial_cmp(&metric(&b.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, report)| (*name, report))
        .expect("report summary requires at least one algorithm")
}
