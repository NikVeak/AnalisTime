from datetime import date

import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf


# прнудительно отключим предупреждения системы
import warnings

warnings.simplefilter(action='ignore', category=Warning)
from pmdarima import auto_arima

warnings.filterwarnings("ignore")


def print_menu():
    print("----------------Анализ временных рядов---------------------------")
    print("-----------------------------------------------------------------")


def main():
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['QuotesDB']
    print(client)
    print_menu()
    run = True
    while run:
        print("1. Продолжить работу")

        print("0. Выход")
        try:
            n = int(input())
            if n == 0:
                print("Завершение работы")
                run = False
                return 0
        except KeyboardInterrupt as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1
        except ValueError as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1

        try:
            path = str(input("Введите путь к файлу (сsv файл): "))
        except KeyboardInterrupt as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1
        except ValueError as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1

        colnames = ['ticker', 'per', 'date', 'time', 'price']
        try:
            read_data = pd.read_csv(path, delimiter=';')
        except FileNotFoundError as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1

        df = pd.DataFrame(read_data)
        df.rename(columns=(
            {
                '<TICKER>': colnames[0],
                '<PER>': colnames[1],
                '<DATE>': colnames[2],
                '<TIME>': colnames[3],
                '<CLOSE>': colnames[4]
            }
        ), inplace=True)
        df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)
        print("Исследуемый набор")
        print(df)
        print("-----------------------------------------------")
        print("")
        print("Оценка стационарности модели по критерию Дики-Фуллера")
        critic_value = 0.05
        adf_test = adfuller(df)
        p_value = adf_test[1]
        print('p-value = ', p_value)
        if p_value > critic_value:
            print("Процесс нестационарный, есть тренд или сезонность")
        else:
            print("Процесс стационарный")

        # print("-----------------------------------------------")
        # plot_acf(df, label="Автокорреляция")
        # plt.show()
        #
        # result_decompose = seasonal_decompose(df, period=int(len(df) / 2))
        # result_decompose.plot()
        # plt.show()
        try:
            stepwise_fit = auto_arima(df, start_p=1, start_q=1,
                                      max_p=3, max_q=3, m=12,
                                      start_P=0, seasonal=True,
                                      d=None, D=1, trace=True,
                                      error_action='ignore',  # we don't want to know if an order does not work
                                      suppress_warnings=True,  # we don't want convergence warnings
                                      stepwise=True)
            print(stepwise_fit.summary())
        except MemoryError as e:
            print(e)
            print("Выход из программы")
            run = False
            return 1

        print("Введите параметры согласно лучшей  модели")
        try:
            print("p = ")
            p = int(input())
            print("q = ")
            q = int(input())
            print("m = ")
            m = int(input())
            print("P = ")
            P = int(input())
            print("D = ")
            D = int(input())
            print("Q = ")
            Q = int(input())
        except KeyboardInterrupt as e:
            print(e)
            print("Выход из программы")
            run = False
        except ValueError as e:
            print(e)
            print("Выход из программы")
            run = False

        model = SARIMAX(df, order=(p, q, m), seasonal_order=(P, D, Q, 12))
        result = model.fit()
        print(result.summary())
        #result.plot_diagnostics(figsize=(15, 12))
        #plt.show()

        print("Введите длину прогноза в днях: ")

        try:
            n_day = int(input())
        except KeyboardInterrupt as e:
            print(e)
            print("Выход из программы")
            run = False
        except ValueError as e:
            print(e)
            print("Выход из программы")
            run = False
        print("Построение прогноза на", n_day, " дней")
        date_res = pd.date_range("22-11-2023", periods=n_day).tolist()
        forecast = result.predict(start=len(df), end=(len(df) - 1) + n_day)
        forecast = pd.DataFrame(forecast)
        print(forecast)
        forecast.to_json(path+"_f"+".json", orient='split', compression = 'infer', index = True)



        forecast['date'] = date_res
        data_forecast = forecast.to_dict('records')
        name_collection = str(input("Введите имя новой коллекции бд: "))
        print(data_forecast)
        collection = db[name_collection].insert_many(data_forecast)

        print(collection)
        forecast.set_index('date', inplace=True)
        print(forecast)
        plt.figure(figsize=(15, 12))
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.title("Прогноз на" + str(n_day) + "дней")
        plt.plot(forecast, color='green')
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()

