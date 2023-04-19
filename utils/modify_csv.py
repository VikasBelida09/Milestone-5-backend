import pandas as pd
import numpy as np

def get_df():
    zoom_csv='../datasets/Zoom-features-2022.xlsx'
    zoom_df=pd.read_excel(zoom_csv,sheet_name=["Dec-2022", "Nov-2022", "Oct-2022", "Sept-2022", "Aug-2022", "July-2022", "June-2022", "May-2022", "April-2022", "March-2022", "Feb-2022", "Jan-2022"])
    merged_df = pd.concat(zoom_df.values(), ignore_index=True)
    merged_df['Release Month']=merged_df['Release Date'].dt.month
    l=merged_df[merged_df['Feature Title'].isnull()].index.tolist()
    merged_df=merged_df.drop(l[0])
    return merged_df

df=get_df()
df.to_csv('zoom.csv',index=False)
