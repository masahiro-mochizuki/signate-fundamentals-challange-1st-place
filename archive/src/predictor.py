# -*- coding: utf-8 -*-
import io
import os
import joblib
import datetime

import numpy as np
import pandas as pd


class ScoringService(object):
    # called automatically in eval time
    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs
    
    @classmethod
    def get_dataset(cls, inputs):
        """ Args:
                inputs (dict[str]): path to dataset files
            Returns:
                dict[pd.DataFrame]: loaded data
        """
        loaders = {
            "stock_labels": lambda fname: pd.read_csv(
                fname, parse_dates=["base_date", "label_date_5", "label_date_10", "label_date_20"]
            ).sort_values(["Local Code", "base_date"]),
            "stock_list": lambda fname: pd.read_csv(
                fname, parse_dates=["Effective Date"],
            ),
            "stock_fin": lambda fname: pd.read_csv(
                fname, parse_dates=[
                    "base_date", "Result_FinancialStatement ModifyDate", "Result_Dividend RecordDate", "Result_Dividend DividendPayableDate", "Forecast_Dividend RecordDate"],
            ).sort_values(["Local Code", "base_date"]),
            "stock_price": lambda fname: pd.read_csv(
                fname,
                parse_dates=["EndOfDayQuote Date", "EndOfDayQuote PreviousCloseDate", "EndOfDayQuote PreviousExchangeOfficialCloseDate"],
            ).sort_values(["Local Code", "EndOfDayQuote Date"]),
        }
        dfs = {k: cls._reduce_mem_usage(v(inputs[k])) for k, v in loaders.items()}
        return dfs

    # called automatically in eval time
    @classmethod
    def get_model(cls, model_path="../model"):
        """Get model method
        Args:
            model_path (str): Path to the trained model directory.
        Returns:
            bool: The return value. True for success, False otherwise.
        """
        cls.model = joblib.load(model_path + "/model.joblib")
        return True

    @classmethod
    def _process_lb_df(cls, lb_df):
        lb_dfs = [lb_df]
        rs = []
        for span in [5, 10, 20]:
            gb = lb_df.groupby("Local Code")
            lb_dfs.append(
                gb[[f"label_date_{span}"]].shift(span).add_prefix("past_"),
            )
            lb_dfs.append(
                gb[[f"label_high_{span}", f"label_low_{span}"]].shift(span).add_prefix("feat_past_"),
            )
        lb_df = pd.concat(lb_dfs, axis=1)
        return lb_df
    
    @classmethod
    def _process_ls_df(cls, ls_df):

        ls_df = pd.concat([
            ls_df,
            ls_df[[
                #Local Code",
                "Section/Products",
                "33 Sector(Code)",
                "17 Sector(Code)",
                "Size Code (New Index Series)"]].astype("category").add_prefix("feat_cat_").astype("category")
            ], axis=1)
        return ls_df
    
    @classmethod
    def _process_fn_df(cls, fn_df):
        fn_df["quarter"] = fn_df["Result_FinancialStatement ReportType"].replace({"Q1": 1, "Q2": 2, "Q3": 3, "Annual":4})

        fp_ym = fn_df["Result_FinancialStatement FiscalPeriodEnd"].str.split("/", expand=True).astype("float32")
        fp_m = (fp_ym[0] - 2015) * 12 + fp_ym[1]
        fn_df["fp_month"] = fp_m

        gb = fn_df[~(fn_df["Result_FinancialStatement FiscalPeriodEnd"].isna())].groupby(["Local Code", "Result_FinancialStatement FiscalPeriodEnd"])
        q_df = gb.nth(-1)
        q_df["quarter_span"] = q_df.groupby(["Local Code"]).fp_month.diff().fillna(3)

        fn_df = fn_df.join(q_df["quarter_span"], on=["Local Code", "Result_FinancialStatement FiscalPeriodEnd"])
        
        year_span = (q_df.groupby(["Local Code", "Result_FinancialStatement FiscalYear"]).quarter_span.mean() * 4).rename("year_span")
        fn_df = fn_df.join(year_span, on=["Local Code", "Result_FinancialStatement FiscalYear"])
        
        cf_cols = [
            "Result_FinancialStatement CashFlowsFromOperatingActivities",
            "Result_FinancialStatement CashFlowsFromFinancingActivities",
            "Result_FinancialStatement CashFlowsFromInvestingActivities",
        ]

        fill_df = fn_df.groupby("Local Code")[cf_cols].ffill().add_prefix("fill_") / (fn_df.year_span.values[:, None] / 12)
        fn_df = pd.concat([fn_df, fill_df], axis=1)

        acc_cols = [
            "Result_FinancialStatement NetSales", "Result_FinancialStatement OperatingIncome",
            "Result_FinancialStatement OrdinaryIncome", "Result_FinancialStatement NetIncome",
        ]

        diff_df = q_df.groupby(["Local Code", "Result_FinancialStatement FiscalYear"])[acc_cols].diff().fillna(
            q_df[acc_cols] / q_df["quarter"].values[:, None]
            )
        
        diff_df = diff_df / (q_df["quarter_span"].values[:, None] / 3)

        ma_df = diff_df.reset_index("Local Code").groupby(["Local Code"])[acc_cols].rolling(4, min_periods=1).mean()
        diff_df = diff_df.add_prefix("diff_")
        ma_df = ma_df.add_prefix("ma_")
        
        fn_df = fn_df.merge(diff_df, on=["Local Code", "Result_FinancialStatement FiscalPeriodEnd"], how="left")
        fn_df = fn_df.merge(ma_df, on=["Local Code", "Result_FinancialStatement FiscalPeriodEnd"], how="left")

        amount_cols = [
            "Result_FinancialStatement TotalAssets", "Result_FinancialStatement NetAssets",
        ]
        amount_cols += ["diff_" + col for col in acc_cols]
        amount_cols += ["ma_" + col for col in acc_cols]
        amount_cols += ["fill_" + col for col in cf_cols]
        amount_cols += acc_cols

        # last year info
        last_q_fn_df = fn_df.groupby(["Local Code", "Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType"])[amount_cols].nth(-1).reset_index()
        last_q_fn_df["Result_FinancialStatement FiscalYear"] = last_q_fn_df["Result_FinancialStatement FiscalYear"] + 1

        ### ratio to same quarter of last year
        key_cols = ["Local Code", "Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType"]
        last_same_q_fn_df = fn_df[key_cols].merge(last_q_fn_df, on=key_cols, how="left")
        last_same_q_rt_df = (fn_df[amount_cols] - last_same_q_fn_df[amount_cols]) / last_same_q_fn_df[amount_cols]
        last_same_q_rt_df = last_same_q_rt_df.add_prefix("feat_ratio_")

        ### ratio from forecast to same quarter of last year
        fc_cols = ["Forecast" + col[6:] for col in acc_cols]
        fc_cols += ["Local Code", "Forecast_FinancialStatement FiscalPeriodEnd"]

        fc_key_cols = ["Local Code", "Forecast_FinancialStatement FiscalYear", "Forecast_FinancialStatement ReportType"]
        last_same_q_fn_df = fn_df[fc_key_cols].rename(columns=lambda x: x.replace("Forecast_","Result_")).merge(last_q_fn_df, on=key_cols, how="left")
        
        fc_df = fn_df[fc_cols].rename(columns=lambda x: "Result" + x[8:])

        last_same_q_fc_rt_df = (fc_df[acc_cols] - last_same_q_fn_df[acc_cols]) / last_same_q_fn_df[acc_cols]
        last_same_q_fc_rt_df = last_same_q_fc_rt_df.add_prefix("feat_fc_ratio_")
        
        fc_quarter = fn_df["Forecast_FinancialStatement ReportType"].replace({"Q1": 1, "Q2": 2, "Q3": 3, "Annual":4})
        fc_quarter_df =  (fc_df[acc_cols] / fc_quarter.values[:, None]).add_prefix("fc_")
        
        cat_cols = [
            "Result_FinancialStatement AccountingStandard", "Result_FinancialStatement ReportType", "Result_FinancialStatement CompanyType",
            #Result_FinancialStatement ChangeOfFiscalYearEnd", 
            "Forecast_FinancialStatement ReportType",
            ]

        fn_df = pd.concat([
            fn_df,
            fn_df[cat_cols].add_prefix("feat_cat_").astype("category"),
            last_same_q_rt_df,
            last_same_q_fc_rt_df,
            fc_quarter_df,
            ], axis=1)

        ### Dividend features ###
        fn_df["fill_Result_Dividend AnnualDividendPerShare"] = fn_df.groupby("Local Code")["Result_Dividend AnnualDividendPerShare"].ffill()
        fn_df["feat_divided_lead"] = ((fn_df["Result_Dividend RecordDate"] - fn_df["base_date"]) / datetime.timedelta(days=1))
        fn_df["feat_fc_divided_lead"] = ((fn_df["Forecast_Dividend RecordDate"] - fn_df["base_date"]) / datetime.timedelta(days=1))
        
        return fn_df
    
    @classmethod
    def _process_pr_df(cls, pr_df):
        
        price_cols = [
            #"EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", "EndOfDayQuote Close",
            "EndOfDayQuote ExchangeOfficialClose", 
            #"EndOfDayQuote VWAP"
            ]
        pr_df[price_cols] = pr_df[price_cols].replace(0.0, np.nan)
        pr_df["diff_high_low"] = (np.log1p(pr_df["EndOfDayQuote High"]) / np.log1p(pr_df["EndOfDayQuote Low"])).astype("float16")
        
        lpr_df = pr_df[price_cols].apply(np.log1p)
        dpr_df = lpr_df.diff()
        dpr_df["Local Code"] = pr_df["Local Code"]
        
        pr_dfs = [pr_df]

        gb = pr_df.groupby("Local Code")
        pr_gb = pr_df.groupby("Local Code")[price_cols]
        dpr_gb = dpr_df.groupby("Local Code")[price_cols]

        f16 = "float16"
        for span in [20, 40, 60, 120]:
            pr_dfs += [
                pr_gb.pct_change(span).add_prefix(f"feat_pc{span}_").astype(f16),
                (lpr_df / dpr_gb.rolling(span, min_periods=1).std().reset_index("Local Code", drop=True)).add_prefix(f"feat_vl{span}_").astype(f16), # volatility
                (pr_df[price_cols] / pr_gb.rolling(span, min_periods=1).mean().reset_index("Local Code", drop=True)).add_prefix(f"feat_rmr{span}_").astype(f16), # ratio to moving average
                gb.diff_high_low.rolling(span, min_periods=1).mean().rename(f"feat_wd{span}").reset_index("Local Code", drop=True).astype(f16),
            ]
        pr_df = pd.concat(pr_dfs, axis=1).rename(columns={"EndOfDayQuote Date": "base_date"})
        pr_df["actual_price"] = pr_df["EndOfDayQuote ExchangeOfficialClose"] * pr_df["EndOfDayQuote CumulativeAdjustmentFactor"]

        pr_df["trading_value"] = pr_df["EndOfDayQuote Volume"] * pr_df["EndOfDayQuote VWAP"].replace(0, np.nan).fillna(pr_df["EndOfDayQuote ExchangeOfficialClose"])
        
        return pr_df
    
    @classmethod
    def _add_pr_ls_feat(cls, pr_df, ls_df):
        ### calculate aggregate market values at a timepoint.
        m_df = pd.merge(
            pr_df[["Local Code", "base_date", "EndOfDayQuote CumulativeAdjustmentFactor"]],
            ls_df.groupby("Local Code")[["Effective Date", "IssuedShareEquityQuote IssuedShare"]].nth(-1).rename(columns={"Effective Date": "base_date"}),
            on=["Local Code", "base_date"],
            )

        m_df["amv_coef"] = m_df["IssuedShareEquityQuote IssuedShare"] * m_df["EndOfDayQuote CumulativeAdjustmentFactor"]
        m_df.drop("base_date", axis=1, inplace=True)
        
        pr_df = pr_df.merge(m_df[["Local Code", "amv_coef"]], on="Local Code", how="left")
        pr_df["EndOfDayQuote VWAP"] = pr_df["EndOfDayQuote VWAP"].replace(0, np.nan).fillna(pr_df["EndOfDayQuote ExchangeOfficialClose"])
        
        price_cols = [
            "EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", "EndOfDayQuote Close", "EndOfDayQuote ExchangeOfficialClose",
            "EndOfDayQuote VWAP",
            ]
        pr_df = pd.concat([pr_df, (pr_df[["amv_coef"]].values * pr_df[price_cols]).add_prefix("amv_")], axis=1)
        
        amv_col = "amv_EndOfDayQuote VWAP"
        pr_df = pr_df.merge(
            pr_df.groupby("base_date")[amv_col].sum().rename("total_amv"),
            on="base_date", how="left")
        pr_df["feat_amv_ratio"] = pr_df[amv_col] / pr_df["total_amv"]
        
        return pr_df
    

    @classmethod
    def _add_final_feats(cls, tb_df):
        amv = tb_df["amv_EndOfDayQuote ExchangeOfficialClose"]
        
        ### fundamental features ###
        tb_df["feat_per"] = amv / tb_df["diff_Result_FinancialStatement NetIncome"]
        tb_df["feat_ma_per"] = amv / tb_df["ma_Result_FinancialStatement NetIncome"]
        tb_df["feat_fc_per"] = amv / tb_df["fc_Result_FinancialStatement NetIncome"]
        
        tb_df["feat_pbr"] = amv / tb_df["Result_FinancialStatement NetAssets"]
        
        tb_df["feat_roe"] = tb_df["diff_Result_FinancialStatement NetIncome"] / tb_df["Result_FinancialStatement NetAssets"]
        tb_df["feat_ma_roe"] = tb_df["ma_Result_FinancialStatement NetIncome"] / tb_df["Result_FinancialStatement NetAssets"]
        tb_df["feat_fc_roe"] = tb_df["fc_Result_FinancialStatement NetIncome"] / tb_df["Result_FinancialStatement NetAssets"]
        
        ### ratio to amv ###
        ra_df = (tb_df[["trading_value", "Result_FinancialStatement NetAssets", "Result_FinancialStatement TotalAssets"]] / amv.values[:, None]).add_prefix("feat_amv_ratio_")
        
        ### divided features ###
        dv_cols = [
            "fill_Result_Dividend AnnualDividendPerShare",
            ]
        dv_df = (tb_df[dv_cols] / tb_df["EndOfDayQuote ExchangeOfficialClose"].values[:, None]).add_prefix("feat_dv_")

        total_dv = tb_df[dv_cols] * tb_df["amv_coef"].values[:, None]

        # divided payout ratio
        dpr_df = (total_dv / tb_df["diff_Result_FinancialStatement NetIncome"].values[:, None]).add_prefix("feat_dpr")
        
        tb_df = pd.concat([tb_df, dv_df, ra_df, dpr_df], axis=1)
        return tb_df
    

    @classmethod
    def get_table(cls, dfs):
        lb_df = dfs["stock_labels"].copy()
        ls_df = dfs["stock_list"].copy()
        fn_df = dfs["stock_fin"].copy()
        pr_df = dfs["stock_price"].copy()

        lb_df = cls._process_lb_df(lb_df)
        ls_df = cls._process_ls_df(ls_df)
        fn_df = cls._process_fn_df(fn_df)
        pr_df = cls._process_pr_df(pr_df)

        pr_df = cls._add_pr_ls_feat(pr_df, ls_df)
        
        # targeted stock
        trg_ls_df = ls_df[ls_df.prediction_target]
        #trg_ls_df = ls_df
        
        # Finalcial Statement for each Q
        fn_df["last_flg"] = fn_df.groupby(["Local Code", "Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType"]).cumcount(ascending=False) == 0

        # Assemble dataframes
        tb_df = lb_df.merge(fn_df, on=["Local Code", "base_date"]).merge(pr_df, on=["Local Code", "base_date"], how="left")
        tb_df = tb_df.merge(trg_ls_df, on="Local Code")
        tb_df = cls._add_final_feats(tb_df)
        tb_df["ID"] = tb_df.base_date.dt.strftime("%Y-%m-%d-") + tb_df["Local Code"].astype("str")
        return tb_df.drop_duplicates("ID", keep="last")
    
    @classmethod
    def _predict(cls, dfs):
        tb_df = cls.get_table(dfs)
        
        transformer = cls.model["transformer"]
        estimators = cls.model["estimators"]
        
        X = transformer.transform(tb_df)
        ps = [estimator.predict(X) for estimator in estimators]
        p = np.dstack(ps).mean(axis=2)
        out_df = pd.DataFrame(p, index=X.index, columns=["high_20", "low_20"])
        out_df.reset_index("ID", inplace=True)
        out_df = out_df[["ID", "high_20", "low_20"]]
        return out_df 
        
    # called automatically in eval time
    @classmethod
    def predict(cls, inputs):
        """Predict method
        Args:
            inputs (dict[str]): paths to the dataset files
        Returns:
            dict[pd.DataFrame]: Inference for the given input.
        """
        dfs = cls.get_dataset(inputs)

        out_df = cls._predict(dfs)

        out = io.StringIO()
        out_df.to_csv(out, header=False, index=False)
        return out.getvalue()

    # the method derived from a code from https://www.kaggle.com/fabiendaniel/elo-world,
    # written by FabienDaniel, which is distributed under the Apache License 2.0:
    # http://www.apache.org/licenses/LICENSE-2.0 
    @staticmethod
    def _reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df