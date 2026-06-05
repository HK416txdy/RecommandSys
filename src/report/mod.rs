use crate::eval::evaluate_algorithm;
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

    let mut out = String::new();
    out.push_str("# 搜索与动态混合推荐系统测试报告\n\n");
    out.push_str("## 数据集与设置\n\n");
    out.push_str(&format!(
        "- 数据格式：MovieLens 100K 兼容 `u.data` / `u.item`\n- 电影数量：{}\n- 评分数量：{}\n- Holdout 比例：{:.2}\n- Top-N：{}\n\n",
        dataset.movies.len(),
        dataset.ratings.len(),
        holdout_ratio,
        top_n
    ));
    out.push_str("## 算法说明\n\n");
    out.push_str("- 基于内容推荐：用电影类型、标题 token、发行年份构建内容向量，并与用户高评分画像计算余弦相似度。\n");
    out.push_str(
        "- 基于知识推荐：从用户高低评分历史抽取偏好类型、排斥类型和年份偏好，再按规则命中程度打分。\n",
    );
    out.push_str("- 用户协同过滤：根据共同评分电影计算用户相似度，并加权预测候选电影评分。\n");
    out.push_str("- 物品协同过滤：根据共同评分用户计算电影相似度，并由用户已评分电影加权预测。\n");
    out.push_str("- 热门度推荐：使用平均评分、评分数量和贝叶斯平滑得到稳定热门度。\n");
    out.push_str("- 矩阵分解：使用 SGD 训练用户/电影隐向量与偏置项。\n");
    out.push_str("- 动态混合推荐：针对每个用户-项目候选，按内容完整度、知识规则命中、CF 覆盖度、热门度稳定性和矩阵分解覆盖度动态归一化权重。\n\n");
    out.push_str("## 指标定义\n\n");
    out.push_str("- 分类准确度：真实评分 >= 4 与预测评分 >= 4 的二分类一致率。\n");
    out.push_str("- nDCG：Top-N 推荐列表对真实相关项目的折损累计增益归一化值。\n");
    out.push_str("- 平均绝对误差 MAE：预测评分与真实评分绝对误差的平均值。\n");
    out.push_str("- 均方根误差 RMSE：预测评分与真实评分平方误差均值的平方根。\n");
    out.push_str("- Top-N 精确度：Top-N 推荐中真实相关项目占比。\n\n");
    out.push_str("## 评测结果\n\n");
    out.push_str("| 算法 | 分类准确度 | nDCG | MAE | RMSE | Top-N 精确度 |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for (algorithm, result) in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            algorithm,
            result.accuracy,
            result.ndcg_at_n,
            result.mae,
            result.rmse,
            result.precision_at_n
        ));
    }
    out.push_str("\n## 测试命令\n\n");
    out.push_str("```powershell\ncargo test\ncargo run -- evaluate --algorithm all --top-n 10 --holdout-ratio 0.2 --report reports/test_report.md\n```\n");
    out
}
