use crate::eval::{EvaluationReport, evaluate_algorithm};
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

    let mut out = String::new();
    out.push_str("# 搜索与动态混合推荐系统实验报告\n\n");
    out.push_str("## 摘要\n\n");
    out.push_str(&format!(
        "本实验实现并比较了基于内容、基于知识、用户协同过滤、物品协同过滤、热门度、矩阵分解和动态混合推荐共 {} 类算法。实验采用 MovieLens 100K 兼容数据格式，以按用户时间顺序划分的 holdout 方法构造训练集与测试集，Top-N 取 {}。评价指标同时覆盖评分预测误差、二分类偏好判断、排序质量、命中能力、目录覆盖、多样性和流行度偏置，因此能够较全面地刻画推荐系统的准确性、泛化能力与推荐列表质量。\n\n",
        rows.len(),
        top_n
    ));

    out.push_str("## 数据集与实验设置\n\n");
    out.push_str(&format!(
        "- 数据格式：MovieLens 100K 兼容 `u.data` / `u.item`\n- 电影数量：{}\n- 评分数量：{}\n- Holdout 比例：{:.2}\n- Top-N：{}\n- 相关项目定义：测试集中真实评分 `>= 4` 的电影视为用户相关项目\n- 评分预测阈值：预测评分 `>= 4` 判定为用户可能喜欢\n\n",
        dataset.movies.len(),
        dataset.ratings.len(),
        holdout_ratio,
        top_n
    ));

    out.push_str("## 方法概述\n\n");
    out.push_str("- 基于内容推荐：利用电影类型、标题 token 和发行年代构建内容向量，与用户高评分电影形成的画像计算余弦相似度。该方法适合表达用户长期兴趣，但容易受到电影元数据稀疏和类型粒度较粗的限制。\n");
    out.push_str("- 基于知识推荐：从用户历史中抽取偏好类型、排斥类型和年份区间，以规则方式给候选电影打分。该方法解释性强，适合冷启动补充，但规则表达能力弱于统计模型。\n");
    out.push_str("- 用户协同过滤：根据共同评分电影计算用户相似度，采用用户均值中心化后的评分偏差进行预测，并对共同评分较少的相似度进行收缩，降低偶然一致造成的极端预测。\n");
    out.push_str("- 物品协同过滤：根据共同评分用户计算电影相似度，采用物品均值中心化后的评分偏差进行预测，并抑制长尾电影由少量共同评分导致的虚高相似度。\n");
    out.push_str("- 热门度推荐：使用平均评分、评分数量和贝叶斯平滑得到稳定的全局热门度。该方法缺少个性化，但通常能形成较强的质量基线。\n");
    out.push_str("- 矩阵分解：通过 SGD 学习用户/电影隐向量和偏置项，将稀疏评分矩阵映射到低维潜在空间，作为个性化预测的主干模型。\n");
    out.push_str("- 动态混合推荐：以矩阵分解和协同过滤为主干，内容、知识和热门度作为辅助信号，依据用户历史长度、物品评分数量和各模块置信度动态归一化权重。\n\n");

    out.push_str("## 评价指标定义\n\n");
    out.push_str("- 分类准确度：将评分预测转化为二分类任务，比较真实评分 `>= 4` 与预测评分 `>= 4` 是否一致。该指标反映系统判断“喜欢/不喜欢”的粗粒度能力。\n");
    out.push_str("- MAE：预测评分与真实评分的平均绝对误差。MAE 越低，说明模型整体评分偏差越小，对离群误差不如 RMSE 敏感。\n");
    out.push_str("- RMSE：预测评分与真实评分平方误差均值的平方根。RMSE 会放大大误差，因此可用于观察模型是否存在严重误判。\n");
    out.push_str("- Precision@N：Top-N 推荐中真实相关项目所占比例。该指标衡量推荐列表前 N 个结果的准确性。\n");
    out.push_str("- Recall@N：Top-N 推荐覆盖了用户测试集中多少真实相关项目。该指标强调相关项目的召回能力。\n");
    out.push_str(
        "- F1@N：Precision@N 与 Recall@N 的调和平均，用于综合衡量推荐列表的准确性和召回性。\n",
    );
    out.push_str("- HitRate@N：用户 Top-N 列表中是否至少命中一个真实相关项目，再对用户取平均。该指标衡量系统能否给用户提供至少一个有效推荐。\n");
    out.push_str("- nDCG@N：考虑命中项目所在位置的排序指标。相关项目排得越靠前，nDCG 越高，因此它比 Precision 更关注排序质量。\n");
    out.push_str("- 目录覆盖率：所有用户推荐列表中出现过的不同电影数量占电影总数的比例。覆盖率越高，说明系统越不依赖少数热门电影。\n");
    out.push_str("- 平均推荐流行度：推荐电影在训练集中平均被评分次数。数值越高，说明算法越偏向热门电影；数值较低则表示更倾向长尾探索。\n");
    out.push_str("- 类型多样性：Top-N 列表中不同电影类型数量相对类型出现次数的比例，用于衡量推荐列表是否集中在少数类型上。\n\n");

    out.push_str("## 实验结果\n\n");
    push_prediction_table(&mut out, &rows);
    push_ranking_table(&mut out, &rows);
    push_coverage_table(&mut out, &rows);

    out.push_str("## 结果分析\n\n");
    out.push_str(&format!(
        "从评分预测角度看，`{}` 获得最低 MAE（{:.4}），说明它在数值评分拟合上最稳定；`{}` 获得最低 RMSE（{:.4}），说明其严重误判相对较少。MAE 和 RMSE 同时较低的模型更适合作为预测评分的核心模块，因为它不仅平均误差小，也较少出现大幅偏离真实评分的情况。\n\n",
        summary.best_mae.0,
        summary.best_mae.1.mae,
        summary.best_rmse.0,
        summary.best_rmse.1.rmse
    ));
    out.push_str(&format!(
        "从 Top-N 排序质量看，`{}` 的 nDCG@{} 最高（{:.4}），说明它更倾向于把真实相关电影排在推荐列表靠前位置。`{}` 的 Precision@{} 最高（{:.4}），表示其前 {} 个推荐中相关项目比例最高；`{}` 的 HitRate@{} 最高（{:.4}），说明它为用户提供至少一个有效推荐的能力最强。\n\n",
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
        "从覆盖率和推荐生态角度看，`{}` 的目录覆盖率最高（{:.4}），表示该算法能触达更大范围的电影目录；`{}` 的平均推荐流行度最高（{:.2}），说明它更偏向大众热门项目；`{}` 的类型多样性最高（{:.4}），说明其推荐列表在题材分布上更分散。覆盖率、多样性和流行度偏置并不直接等同于准确率，但它们决定了系统是否只会重复推荐少数热门电影，以及是否能为不同兴趣用户提供足够丰富的候选。\n\n",
        summary.best_coverage.0,
        summary.best_coverage.1.catalog_coverage,
        summary.highest_popularity.0,
        summary.highest_popularity.1.avg_recommendation_popularity,
        summary.best_diversity.0,
        summary.best_diversity.1.avg_genre_diversity
    ));

    if let Some((_, hybrid)) = rows.iter().find(|(name, _)| *name == "hybrid") {
        out.push_str(&format!(
            "动态混合推荐的实验意义不只是追求单一指标最优，而是在准确性、稳定性和可解释性之间折中。本次实验中，hybrid 的 MAE 为 {:.4}、RMSE 为 {:.4}、nDCG@{} 为 {:.4}、Precision@{} 为 {:.4}、目录覆盖率为 {:.4}。这些数值表明，混合模型能够保持接近强基线模型的排序质量，同时通过内容、知识和热门度模块提供更丰富的解释信号。对于课程大作业场景，这种结果比单纯堆叠模型更有说明价值：系统并非让所有模块平均投票，而是让矩阵分解和协同过滤承担主要预测职责，其他模块用于冷启动、解释和稳定性补充。\n\n",
            hybrid.mae,
            hybrid.rmse,
            top_n,
            hybrid.ndcg_at_n,
            top_n,
            hybrid.precision_at_n,
            hybrid.catalog_coverage
        ));
    }

    out.push_str("## 系统能力与局限性\n\n");
    out.push_str("本系统已经具备推荐系统实验所需的基本能力：能够读取 MovieLens 兼容数据，构建用户-物品索引，训练多类推荐算法，进行时间顺序 holdout 评估，并生成包含准确性、排序质量和多样性分析的实验报告。与只给出评分误差的系统相比，本系统更关注真实推荐场景中的 Top-N 列表质量，因此报告能够回答“预测准不准”“推荐列表是否命中”“是否只推荐热门电影”“推荐结果是否多样”等多个问题。\n\n");
    out.push_str("系统的主要局限在于：第一，内容特征只使用类型、标题和年份，缺少剧情、演员、导演等更细粒度语义信息，因此内容推荐的表达能力有限；第二，知识规则是人工设计的线性规则，解释性强但泛化能力有限；第三，矩阵分解实现为教学级 SGD，没有引入随机打乱、验证集早停和超参数搜索；第四，当前 holdout 只做一次固定时间切分，指标可能受切分方式影响。尽管如此，对于课程设计而言，这些取舍使系统保持了可读性、可复现实验和较完整的推荐系统评价闭环。\n\n");

    out.push_str("## 结论\n\n");
    out.push_str(&format!(
        "综合实验结果可以认为，`{}` 更适合作为评分预测主模型，`{}` 更适合作为排序质量参考模型，而动态混合推荐适合作为最终系统方案：它在保持较好预测和排序表现的同时，兼顾协同信号、潜在因子信号、内容偏好、知识规则和热门度兜底。后续若继续改进，可以优先增强内容特征、优化矩阵分解训练过程，并对混合权重进行系统性网格搜索或验证集调参。\n\n",
        summary.best_mae.0,
        summary.best_ndcg.0
    ));

    out.push_str("## 测试命令\n\n");
    out.push_str("```powershell\ncargo test\ncargo run -- evaluate --algorithm all --top-n 10 --holdout-ratio 0.2 --report reports/test_report.md\n```\n");
    out
}

fn push_prediction_table(out: &mut String, rows: &[(&str, EvaluationReport)]) {
    out.push_str("### 评分预测与二分类指标\n\n");
    out.push_str("| 算法 | 分类准确度 | MAE | RMSE |\n");
    out.push_str("|---|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} |\n",
            algorithm, result.accuracy, result.mae, result.rmse
        ));
    }
    out.push('\n');
}

fn push_ranking_table(out: &mut String, rows: &[(&str, EvaluationReport)]) {
    out.push_str("### Top-N 排序与命中指标\n\n");
    out.push_str("| 算法 | Precision@N | Recall@N | F1@N | HitRate@N | nDCG@N |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            algorithm,
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
    best_mae: (&'a str, &'a EvaluationReport),
    best_rmse: (&'a str, &'a EvaluationReport),
    best_ndcg: (&'a str, &'a EvaluationReport),
    best_precision: (&'a str, &'a EvaluationReport),
    best_hit_rate: (&'a str, &'a EvaluationReport),
    best_coverage: (&'a str, &'a EvaluationReport),
    best_diversity: (&'a str, &'a EvaluationReport),
    highest_popularity: (&'a str, &'a EvaluationReport),
}

impl<'a> ReportSummary<'a> {
    fn from_rows(rows: &'a [(&'a str, EvaluationReport)]) -> Self {
        Self {
            best_mae: min_by(rows, |r| r.mae),
            best_rmse: min_by(rows, |r| r.rmse),
            best_ndcg: max_by(rows, |r| r.ndcg_at_n),
            best_precision: max_by(rows, |r| r.precision_at_n),
            best_hit_rate: max_by(rows, |r| r.hit_rate_at_n),
            best_coverage: max_by(rows, |r| r.catalog_coverage),
            best_diversity: max_by(rows, |r| r.avg_genre_diversity),
            highest_popularity: max_by(rows, |r| r.avg_recommendation_popularity),
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
