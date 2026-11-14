# -*- coding: utf-8 -*-
"""
任务 5.1 + 5.2：建立并校验广西 146 个泥石流点综合分析数据库

输入：
    第三轮_空间匹配验收/广西146个泥石流点_空间匹配结果.csv
    第四轮_降水匹配/广西146个泥石流点_降水提取结果_2001-2020.csv
输出：
    广西146个泥石流点_综合分析数据库.csv
    广西146个泥石流点_综合分析数据库.gpkg
    00_校验记录.xlsx

作者：TraeCode    日期：2026-11-14
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

REPO = r"C:\Users\sanhua\Documents\debris-flow-for-GX"
F3 = os.path.join(REPO, r"第三轮_空间匹配验收\广西146个泥石流点_空间匹配结果.csv")
F4 = os.path.join(REPO, r"第四轮_降水匹配\广西146个泥石流点_降水提取结果_2001-2020.csv")
OUT = r"C:\Users\sanhua\Documents\Trae-CN\temp_20261112_01\综合分析数据库"
os.makedirs(OUT, exist_ok=True)

KEY = "灾害体编号"
DUP = ["灾害体名称", "市", "县", "经度", "纬度"]      # 第四轮中与第三轮重复的列（备注改名为备注_降水保留）

# 原始属性（来自灾害点矢量库，必须原样保留）
ORIG_COLS = ["灾害体编号", "灾害体名称", "灾害体类型", "市", "县", "乡镇", "地理位置",
             "经度", "纬度", "点类型", "险情等级", "灾害等级", "威胁人口", "威胁财产",
             "坡顶最大标高", "坡底最小标高", "目前稳定程度", "监测建议", "Guid"]
# 核心分析因子（进入建模）
CORE_COLS = ["ELEV_M", "SLOPE_DEG", "RELIEF_M", "DIST_RIV_M",
             "PREC_MM_YR", "PREC_WS_MM", "WS_RATIO"]
# 辅助/参考字段（保留但不进核心分析）
REF_COLS = ["GEO_AGE", "GEO_AGE_FULL", "LANDFORM", "RELIEF_CLASS",
            "TYPE_RAW", "GEO_UNIT", "RIV_NAME"]


def read(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={KEY: str})
    df[KEY] = df[KEY].str.strip()
    return df


def main():
    third = read(F3)
    fourth = read(F4)

    # ---------------- 合并前检查
    assert third[KEY].is_unique, "第三轮编号不唯一"
    assert fourth[KEY].is_unique, "第四轮编号不唯一"
    assert len(third) == 146 and len(fourth) == 146
    set3, set4 = set(third[KEY]), set(fourth[KEY])
    assert set3 == set4, f"编号集合不一致：仅第三轮 {set3-set4}；仅第四轮 {set4-set3}"

    # 坐标一致性（两轮同名点必须同位）
    coord = third[[KEY, "经度", "纬度"]].merge(fourth[[KEY, "经度", "纬度"]], on=KEY,
                                            suffixes=("_3", "_4"))
    dlon = (coord["经度_3"] - coord["经度_4"]).abs()
    dlat = (coord["纬度_3"] - coord["纬度_4"]).abs()
    coord_ok = bool((dlon < 1e-9).all() and (dlat < 1e-9).all())

    # ---------------- 合并
    drop_cols = [c for c in DUP if c in fourth.columns]
    f4 = fourth.rename(columns={"备注": "备注_降水"}).drop(columns=drop_cols)
    final = third.merge(f4, on=KEY, how="left", validate="one_to_one")
    assert len(final) == 146, f"合并后记录数 {len(final)}"

    # ---------------- 列顺序整理
    core = [c for c in ORIG_COLS]
    tail = [c for c in final.columns if c not in core]
    # 让核心因子紧跟原始属性
    ordered = core + [c for c in CORE_COLS if c in final.columns] + \
              [c for c in tail if c not in CORE_COLS]
    final = final[ordered]

    # ---------------- 数值类型
    for c in ["经度", "纬度", "ELEV_M", "SLOPE_DEG", "RELIEF_M", "DIST_RIV_M",
              "PREC_MM_YR", "PREC_ANN_3X3", "PREC_DIFF_PCT", "PREC_WS_MM",
              "PREC_WS_3X3", "WS_RATIO"]:
        if c in final.columns:
            final[c] = pd.to_numeric(final[c], errors="coerce")
    for c in ["PIX_ROW", "PIX_COL", "DEM_HIT", "GEO_MATCH", "LF_MATCH"]:
        if c in final.columns:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype("Int64")

    # ---------------- 任务 5.2 校验
    checks = []

    def chk(item, std, actual, ok):
        checks.append([item, std, actual, "通过" if ok else "不通过"])

    chk("总记录数", "146", f"{len(final)}", len(final) == 146)
    chk("唯一编号", "146", f"{final[KEY].nunique()}", final[KEY].nunique() == 146)
    for c, cn in [("PREC_MM_YR", "年降水"), ("PREC_WS_MM", "汛期降水"),
                  ("ELEV_M", "高程"), ("SLOPE_DEG", "坡度"),
                  ("RELIEF_M", "起伏度"), ("DIST_RIV_M", "河流距离")]:
        n = int(final[c].isna().sum())
        chk(f"{cn}空值（{c}）", "0", f"{n}", n == 0)
    missing_orig = [c for c in ORIG_COLS if c not in final.columns]
    chk("原始属性完整", "19 个原始字段全在", f"缺失 {missing_orig if missing_orig else '无'}", not missing_orig)
    vc = final["点类型"].value_counts().to_dict()
    chk("45/101 点类型保留", "在册隐患点 45 / 非隐患点 101",
        f"在册隐患点 {vc.get('在册隐患点', 0)} / 非隐患点 {vc.get('非隐患点', 0)}",
        vc.get("在册隐患点", 0) == 45 and vc.get("非隐患点", 0) == 101)
    n_type = int(final["TYPE_RAW"].fillna("").astype(str).str.strip().ne("").sum())
    chk("岩性字段保留", "保留原始记录，不进入核心分析",
        f"TYPE_RAW 非空 {n_type}/146（记录保留，仅作参考）", "TYPE_RAW" in final.columns)
    chk("坐标一致性", "两轮经纬度完全一致", f"最大差值 {max(dlon.max(), dlat.max()):.2e}°", coord_ok)
    chk("合并方式", "按灾害体编号 one_to_one", "merge validate='one_to_one' 通过", True)
    chk("核心因子完整", "7 个核心字段无缺失",
        f"缺失 {int(final[CORE_COLS].isna().sum().sum())} 个", int(final[CORE_COLS].isna().sum().sum()) == 0)

    # ---------------- 写出 CSV
    csv_out = os.path.join(OUT, "广西146个泥石流点_综合分析数据库.csv")
    final.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print("写出", csv_out, final.shape)

    # ---------------- 写出 GeoPackage
    gdf = gpd.GeoDataFrame(final.copy(),
                           geometry=gpd.points_from_xy(final["经度"], final["纬度"]),
                           crs="EPSG:4326")
    gpkg_out = os.path.join(OUT, "广西146个泥石流点_综合分析数据库.gpkg")
    if os.path.exists(gpkg_out):
        os.remove(gpkg_out)
    gdf.to_file(gpkg_out, layer="泥石流点综合分析", driver="GPKG")
    print("写出", gpkg_out)

    # ---------------- 写出校验记录
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "00_校验记录"
    ws.append(["广西 146 个泥石流点 · 综合分析数据库校验"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append(["合并字段：灾害体编号；合并方式：left + validate=one_to_one"])
    ws.append([])
    ws.append(["检查项目", "通过标准", "实测结果", "结论"])
    for i, c in enumerate(ws[4], start=1):
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r in checks:
        ws.append(r)
    ws.append([])
    ws.append(["核心分析因子（7 个）", "、".join(CORE_COLS)])
    ws.append(["参考字段（不进核心分析）", "、".join(REF_COLS)])
    ws.append(["说明", "TYPE_RAW（岩性）仅 39/294 个地质面有值，保留但不参与建模"])
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 48
    ws.column_dimensions["D"].width = 10
    ws.freeze_panes = "A5"
    xlsx_out = os.path.join(OUT, "00_校验记录.xlsx")
    wb.save(xlsx_out)
    print("写出", xlsx_out)

    # ---------------- 控制台汇总
    print("\n校验结果：")
    for r in checks:
        print(f"  [{r[3]}] {r[0]}: {r[2]}")
    bad = [r for r in checks if r[3] != "通过"]
    print(f"\n不通过项 {len(bad)} 个")
    print("\n核心因子描述统计：")
    print(final[CORE_COLS].describe().round(2).to_string())

    return dict(n=len(final), cols=final.shape[1], bad=len(bad))


if __name__ == "__main__":
    main()
