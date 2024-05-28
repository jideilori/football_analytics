import pandas as pd

def create_and_write_csv(file_name):
    # Define the column names
    columns = ['frame_count','id', 'x', 'y',  'color']
    
    # Create an empty DataFrame with these column names
    df = pd.DataFrame(columns=columns)
    
    # Save the empty DataFrame to a CSV file
    df.to_csv(file_name, index=False)
    
    # Function to add a row to the CSV
    def add_row(frame_count,id, x, y, color):
        # Create a new row as a dictionary
        row = {
            'frame_count':frame_count,
            'id': id,
            'x': x,
            'y': y,
            'color': color
        }
        
        # Append the row to the DataFrame
        df_new = pd.DataFrame([row])
        
        # Append the new DataFrame to the CSV file
        df_new.to_csv(file_name, mode='a', header=False, index=False)
    
    # Return the function to add rows
    return add_row


