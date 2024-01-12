import streamlit as st
import pandas as pd

st.title("Ближайшие объекты к Земле")

st.header("О датасете")

st.write("Этот файл содержит различные параметры/характеристики, на основании которых конкретный астероид, который уже классифицируется как ближайший земной объект, может быть или не быть опасным.")

df = pd.read_csv("csgo_task.csv")
st.table(df.head())

st.write("Всего в датасете 116693 объектов и 16 признаков")

st.markdown("""
    ## Описание Датасета CS:GO
    **Файл датасета:** `csgo_task.csv`

    **Описание:**
    Данный датасет содержит статистическую информацию о матчах в популярной компьютерной игре Counter-Strike: Global Offensive (CS:GO). Включает следующие столбцы:

    - `index`: Индекс записи.
    - `time_left`: Время до конца раунда.
    - `ct_score`: Счёт команды контр-террористов.
    - `t_score`: Счёт команды террористов.
    - `ct_health`: Общее здоровье команды контр-террористов.
    - `t_health`: Общее здоровье команды террористов.
    - `ct_armor`: Уровень брони команды контр-террористов.
    - `t_armor`: Уровень брони команды террористов.
    - `ct_money`: Деньги команды контр-террористов.
    - `t_money`: Деньги команды террористов.
    - `ct_helmets`: Шлемы команды контр-террористов.
    - `t_helmets`: Шлемы команды террористов.
    - `ct_defuse_kits`: Комплекты для обезвреживания бомбы у CT.
    - `ct_players_alive`: Живые игроки команды контр-террористов.
    - `t_players_alive`: Живые игроки команды террористов.
    - `bomb_planted_True`: Индикатор заложенной бомбы.
                
    **Особенности предобработки данных:**
    - Удаление лишних столбцов, например, 'index'.
    - Обработка пропущенных значений.
    - Нормализация числовых данных для улучшения производительности моделей.
    - Кодирование категориальных переменных.
    """)