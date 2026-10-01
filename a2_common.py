"""
COMP90049 A2 — 共用模块（数据层交付给建模层的接口）

目的：让四个模型跑在**完全相同**的数据划分与预处理上。
评分表 Validation design 满分档要求「所有模型用同一套验证方法」，
各自写各自的 pipeline 会导致比较不公平，直接丢分。

用法
----
    import a2_common as A2

    df = A2.load_table(24)                     # 第 24 天的宽表
    tr, va, te = A2.make_splits(df)            # 按 id_student 分组 + 分层，60/20/20
    A2.describe_splits(tr, va, te)             # 打印各集合样本数与类别比例（报告 Method 要用）

    cols = A2.feature_cols("engineered")       # 或 "baseline"
    prep = A2.make_preprocessor(cols, log_transform=True)

    pipe = Pipeline([("prep", prep), ("clf", YourModel())])
    pipe.fit(tr[cols], tr[A2.TARGET])
    scores = A2.evaluate(va[A2.TARGET], pipe.predict(va[cols]),
                         pipe.predict_proba(va[cols])[:, 1])

约定
----
* 调参只看验证集 (va)，测试集 (te) 只在最后评估一次
* 任何模型都不要自己另做划分或预处理
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- 常量
SEED = 42
RANDOM_SEED = SEED          # 旧名，保留以免别处引用报错
DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

KEY = ["code_module", "code_presentation", "id_student"]
GROUP_COL = "id_student"
TARGET = "at_risk"          # 1 = 未完成课程 (Fail/Withdrawn)，0 = 完成 (Pass/Distinction)
CUTOFF_DAYS = [0, 24, 60, 120]
MAIN_CUTOFF = 24            # 主结果用第 24 天（第一次作业已交）

_manifest = None


def manifest():
    """读 manifest.json（缓存）。"""
    global _manifest
    if _manifest is None:
        with open(PROCESSED_DIR / "manifest.json", encoding="utf-8") as fh:
            _manifest = json.load(fh)
    return _manifest


def feature_cols(which="engineered"):
    """'baseline' = 仅背景特征；'engineered' = 背景 + 行为特征。"""
    return list(manifest()["feature_sets"][which])


def load_table(cutoff=MAIN_CUTOFF):
    """读某个截断点的宽表。"""
    path = PROCESSED_DIR / f"wide_cutoff_{cutoff:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，请先跑 A2_01_clean_and_aggregate.ipynb")
    return pd.read_csv(path)


# ---------------------------------------------------------------- 划分
def make_splits(df, seed=SEED):
    """按 id_student 分组 + 按 at_risk 分层，划分 60/20/20。

    为什么分组：3,538 名学生出现于多条记录（最多 5 条）。随机划分会让同一学生
    同时落入训练集与测试集，模型可凭记住个体特征作弊，使测试分数虚高。
    实证：第一次完成者的第二次完成率 55.9%，第一次未完成者 24.7%，相差 31 个百分点。

    为什么同时分层：保持三个集合的类别比例与原数据一致，使比较有意义。
    """
    y, g = df[TARGET], df[GROUP_COL]

    # 5 折取 1 折作测试集 = 20%
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_pos, test_pos = next(outer.split(df, y, g))
    trainval = df.iloc[trainval_pos].reset_index(drop=True)
    test = df.iloc[test_pos].reset_index(drop=True)

    # 剩下 80% 再按 4 折取 1 折作验证集 = 总量的 20%
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_pos, val_pos = next(inner.split(trainval, trainval[TARGET], trainval[GROUP_COL]))
    train = trainval.iloc[tr_pos].reset_index(drop=True)
    val = trainval.iloc[val_pos].reset_index(drop=True)

    # 硬性校验：任何学生都不能跨集合出现
    s_tr, s_va, s_te = (set(d[GROUP_COL]) for d in (train, val, test))
    assert not (s_tr & s_va), "训练集与验证集有学生重叠"
    assert not (s_tr & s_te), "训练集与测试集有学生重叠"
    assert not (s_va & s_te), "验证集与测试集有学生重叠"

    return train, val, test


def describe_splits(train, val, test):
    """打印划分统计 —— 报告 Method 3.3 要引用这张表。"""
    rows = []
    total = len(train) + len(val) + len(test)
    for name, d in [("train", train), ("validation", val), ("test", test)]:
        rows.append({
            "split": name,
            "n": len(d),
            "占比%": round(len(d) / total * 100, 1),
            "唯一学生": d[GROUP_COL].nunique(),
            "正类(at_risk=1)%": round(d[TARGET].mean() * 100, 2),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\n✅ 三个集合之间无学生重叠（已在 make_splits 中断言）")
    return out


# ---------------------------------------------------------------- 预处理
def _log1p_nonneg(X):
    """log(1+x)，先裁掉负值保证定义域。具名函数而非 lambda，便于序列化。"""
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(cols, log_transform=False):
    """构造预处理流水线。

    数值列：median 填补 → （可选 log1p）→ 标准化
    类别列：most_frequent 填补 → one-hot

    ⚠️ 必须放在 Pipeline 里，只对训练集 fit。
    若先对全量数据标准化再划分，测试集的统计信息会渗入训练，属于 leakage，
    评分表 Validation design 直接掉到 0.5 分档。

    log_transform=True 时，只对 manifest 的 skewed_cols 施加 log1p。
    这些列偏度 > 2 且 log 后降幅 ≥ 50%（如 vle_max_clicks_day: 31.66 → -0.61）。
    注意 vle_prop_* 这类 0-1 占比列虽然偏度也高，但那是稀疏而非长尾，log 无效，故未纳入。
    """
    mf = manifest()
    cat = [c for c in cols if c in mf["categorical_features"]]
    num = [c for c in cols if c not in cat]

    skewed = [c for c in num if c in mf.get("skewed_cols", [])] if log_transform else []
    plain = [c for c in num if c not in skewed]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skew_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", num_pipe, plain)]
    if skewed:
        transformers.append(("skew", skew_pipe, skewed))
    if cat:
        transformers.append(("cat", cat_pipe, cat))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------- 评估
def evaluate(y_true, y_pred, y_proba=None):
    """统一的评估指标。正类 = 1 = 未完成课程（高风险）。

    指标选择理由（报告 Method 3.5 要写）：
      accuracy   直观，但类别不平衡时会高估，仅作参考
      precision  预警了的学生里真有风险的比例 —— 干预资源有限，误报浪费人力
      recall     真有风险的学生里被预警到的比例 —— 漏掉一个的代价高于误报一个
      macro_f1   各类 F1 取平均，不受类别大小影响，不平衡下比 accuracy 公正
      roc_auc    不依赖判定阈值，便于跨模型比较
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: round(float(v), 4) for k, v in out.items()}


def confusion(y_true, y_pred):
    """混淆矩阵，带可读的行列名。"""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm,
                        index=["实际: 完成(0)", "实际: 未完成(1)"],
                        columns=["预测: 完成(0)", "预测: 未完成(1)"])


def score_model(model, cols, train, val, log_transform=False, fit_on=None):
    """一行跑完：建 pipeline → 训练 → 在验证集上评估。返回 (pipeline, 指标 dict)。"""
    pipe = Pipeline([("prep", make_preprocessor(cols, log_transform)), ("clf", model)])
    fit_df = train if fit_on is None else fit_on
    pipe.fit(fit_df[cols], fit_df[TARGET])
    pred = pipe.predict(val[cols])
    proba = pipe.predict_proba(val[cols])[:, 1] if hasattr(pipe, "predict_proba") else None
    return pipe, evaluate(val[TARGET], pred, proba)
