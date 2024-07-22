import mysql.connector
import pandas as pd

db = mysql.connector.connect(user='root',password='abc123',host='127.0.0.1',port=3306,database='bookstore')
ramka = pd.read_sql('SELECT * FROM books',con=db)

print(ramka)
