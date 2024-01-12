import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

df = pd.read_csv("new_csgo.csv")

df_old = pd.read_csv("csgo_task.csv")

st.title("Разведывательный анализ данных")

st.subheader("Выведем основные статистические данные")

st.table(df.describe())

variable_1 = st.sidebar.radio('Выберите первый признак для точечного графика', (df.drop("bomb_planted", axis=1).columns))
variable_2 = st.sidebar.radio('Выберите второй признак для точечного графика', (df.drop("bomb_planted", axis=1).columns), 
index=2)

st.subheader("Точечный график")

fig2, ax2 = plt.subplots()
ax2.scatter(df[variable_1], df[variable_2])
plt.xlabel(variable_1)
plt.ylabel(variable_2)
st.pyplot(fig2)

st.subheader("Гистограммы")

variable_3 = st.sidebar.radio('Выберите признак для гистограммы', (df.drop("bomb_planted", axis=1).columns))

new_df = df[variable_3]
fig1, ax1 = plt.subplots()
plt.title(f"{variable_3}")
ax1.hist(new_df)
st.pyplot(fig1)

variable_4 = st.sidebar.radio('Выберите признак для круговой диаграммы', ("ct_score","t_score","map","bomb_planted","ct_helmets","t_helmets","ct_defuse_kits",'ct_players_alive','t_players_alive'))

new_df = np.unique(df[variable_4], return_counts=True)
fig3, ax3 = plt.subplots()
st.subheader("Круговая диаграмма")
plt.title(f"{variable_4}")
ax3.pie(new_df[1],labels=df[variable_4].unique().tolist(), autopct='%1.1f%%')
st.pyplot(fig3)

st.subheader("Тепловая карта датасета")


fig4, ax4 = plt.subplots()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".1f")
st.pyplot(fig4)
