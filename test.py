from finrl.config import (
    TECHNICAL_INDICATORS_LIST,
    TEST_END_DATE,
    TEST_START_DATE,
    RLlib_PARAMS,
)

from finrl.config_tickers import DOW_30_TICKER



from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
def test(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        env,
        model_name,
        if_vix=True,
        **kwargs
):
    # import DRL agents
    from finrl.agents.risk.models import DRLAgent as DRLAgent_sb3


    # import data processor
    from finrl.finrl_meta.data_processor import DataProcessor

    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2 ** 7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))


    if drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )

        return episode_total_assets
    elif drl_lib == "risk":
        from finrl.agents.risk.models import DRLAgent as DRLAgent2
        episode_total_assets = DRLAgent2.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = {}  # in current finrl_meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="risk",
        env=env,
        model_name="ppo",
        cwd="./saved_models/results/1111238riskppoyahoofinance/mode",
        net_dimension=512,
        kwargs=kwargs,
    )
