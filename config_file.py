import numpy as np
import pandas as pd
import glob
from statistics import mean

class Config:
    def __init__(self):
        # change local path
        path = r"C:\Users\matan\technion\semester6\ML_project\SapProject\Data\Surfers"
        self.filenames = glob.glob(path + "/*", recursive=True)
        filenames_from_dict = []
        dicts = []
        for filename in self.filenames:
            if not filename.endswith(".csv"):
                filenames_from_dict.append(glob.glob(filename + "/*", recursive=True))
                dicts.append(filename)
        self.filenames = [file for file in self.filenames if file.endswith(".csv")]
        for fi in filenames_from_dict:
            self.filenames.extend(fi)

        # for loop to iterate all excel files
        for file in self.filenames:
            # print("Reading file = ",file)
            # print(pd.read_csv(file))
            pass

    def all_trims_list(self):
        df = pd.DataFrame()
        for file in self.filenames:
            label_name = file[:-4]
            label_name = label_name.split('/')[-1]
            tmp_df = pd.read_csv(file)
            df[label_name] = tmp_df['Trim Fore / Aft']
        return df.T

    def all_speed_df(self):
        df = pd.DataFrame(columns=['max speed','avg speed','heel segment'])
        for file in self.filenames:
            label_name = file[:-4]
            label_name = label_name.split('/')[-1]
            tmp_df = pd.read_csv(file)
            self.calc_all_heel_segment(tmp_df)
            for i,seg in tmp_df.groupby(['Heel Segment']).iterrows():
                print(seg)
                val = tmp_df.loc[tmp_df['Heel Segment']==seg[0]]['SOG - Speed over Ground']
                df[f'{label_name}_{seg[0]}'] = df.append({max(val), mean(val), seg[0]})
        return df

    def all_paths_list(self):
        return self.filenames

    def calc_single_heel_segment(self, wind_direction, boat_heading, heel_angle):
        #   https://ibb.co/2dNmsyB
        # Determine the angle between the wind direction and the boat heading
        angle_between = abs(wind_direction - boat_heading)
        if angle_between > 180:
            angle_between = 360 - angle_between

        # Determine the point of sail based on the angle between wind direction and boat heading
        if angle_between <= 45:
            point_of_sail = 'close-hauled'
        elif angle_between <= 135:
            point_of_sail = 'beam reach'
        elif angle_between <= 225:
            point_of_sail = 'broad reach'
        else:
            point_of_sail = 'running'
        # Determine whether the boat is sailing upwind or downwind based on point of sail and heel angle
        if point_of_sail == 'close-hauled': #  0 <= angle_between <= 45
            return 'upwind'
        elif point_of_sail == 'running': #  135 <= angle_between
            return 'downwind'
        else:
            return 'indeterminate'

    def calc_all_heel_segment(self, df):
        df['Heel Segment'] = np.NAN
        for i, row in df.iterrows():
            if "TWD - True Wind Direction" not in row.index:
                segment = 'indeterminate'
            else:
                segment = self.calc_single_heel_segment(row["TWD - True Wind Direction"],
                                                            row['HDT - Heading True'],
                                                            row['Heel'])
            df.at[i,'Heel Segment'] = segment
# %%
