import pandas as pd
import sqlite3

# Step 1: Load the provided CSV into a DataFrame
csv_file = 'Cleaned_Feedback_Data.csv'
table_name = 'feedback_table'
df = pd.read_csv(csv_file)

# Step 2: Create SQL INSERT statements with all columns as TEXT
sql_file = 'output.sql'
with open(sql_file, 'w') as file:
    # Write CREATE TABLE statement
    columns = ', '.join([f'"{col}" TEXT' for col in df.columns])  # Quotes for column names
    file.write(f'CREATE TABLE IF NOT EXISTS {table_name} ({columns});\n\n')

    # Write INSERT statements
    for _, row in df.iterrows():
        # Escape single quotes in values
        values = ', '.join(["'" + str(value).replace("'", "''") + "'" for value in row])
        file.write(f'INSERT INTO {table_name} VALUES ({values});\n')

print(f"SQL file '{sql_file}' created successfully!")

# Step 3: Populate SQLite database
db_file = 'feedback_database.db'
connection = sqlite3.connect(db_file)

# Execute the SQL script
with open(sql_file, 'r') as file:
    sql_script = file.read()

cursor = connection.cursor()
cursor.executescript(sql_script)
connection.commit()

print(f"Data inserted into database '{db_file}' successfully!")

# Step 4: Read and print the DataFrame from the database
query = f"SELECT * FROM {table_name};"
df_from_db = pd.read_sql_query(query, connection)

# Close the connection
connection.close()

# Display the DataFrame
print("Data retrieved from the database:")
print(df_from_db)
