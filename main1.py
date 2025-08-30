import numpy as np
import pandas as pd
import fastf1
import os
import tabulate as tb
import datetime

from concurrent.futures import ThreadPoolExecutor
from fastf1.events import get_event_schedule

CACHE_DIR = './cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR) # learning-use a custom directory or default


# list of races and calender
class F1calender():
    
    def main_calender(self,year_):
        df = fastf1.get_event_schedule(year_)
        expected_columns = ['RoundNumber', 'EventName', 'Country', 'Location', 'EventDate', 'QualifyingDate', 'SprintDate']
        available_columns = [col for col in expected_columns if col in df.columns]
        df = df[available_columns]
        rename_map = {
            'RoundNumber': 'ROUND','EventName': 'EVENT','Country': 'COUNTRY',
            'Location': 'LOCATION','EventDate': 'RACE DAY','QualifyingDate': 'QUALI','SprintDate': 'SPRINT'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("\n📅 F1 Calendar 📅")
        print(tb.tabulate(df, headers='keys', tablefmt="fancy_grid"))
    
    def countdown_to_next_race(year):
        df = fastf1.get_event_schedule(year)
        now = datetime.datetime.now()
        future_races = df[df['EventDate'] > now].sort_values(by='EventDate')
        if future_races.empty:
            print("🏁 No upcoming races this season.")
            return
        next_race = future_races.iloc[0]
        race_name = next_race['EventName']
        race_date = next_race['EventDate']
        countdown = race_date - now
        print(f"\n⏳ Next race: {race_name} on {race_date.strftime('%A, %d %B %Y - %H:%M')}")
        print(f"🕒 Time left: {countdown.days} days, {countdown.seconds // 3600} hours, {(countdown.seconds % 3600) // 60} minutes")


class f1teamsession():
    def __init__(self, year, session, round_):
        self.year = year
        self.session = session  # e.g., 'Q', 'R'
        self.round = round_
        self.get_session_results()
        
    def get_session_results(self):
        session = fastf1.get_session(self.year, self.round, self.session)
        self.session = session.load()
        self.results = session.results
        print("\n📊 Session Results 📊")
        print(self.results)
        
    def indivdual_session_results(self):
        df = pd.DataFrame(self.results)
        result_dict = df.to_dict(orient="records")
        driver_to_find = input("ENTER DRIVER'S NAME::").strip().lower()
        flag = 0
        for drv in result_dict:
            if drv["FullName"].strip().lower() == driver_to_find:
                print(f"\n👤 Driver Results: {drv['FullName']} 👤")
                print(drv)
                return
        if flag == 0:
            print("NOT FOUND")
    
    def top_3_of_particluar_session(self):
        df = self.results[['Position', 'FullName', 'TeamName', 'Time']]
        df = df.sort_values(by="Position")
        top3 = df.head(3).reset_index(drop=True)

        print("\n🏆 Top 3 Drivers of Session 🏆")
        print(tb.tabulate(top3, headers='keys', tablefmt='fancy_grid'))

    def get_team_summary(self):
        te_m = input("ENTER TEAM TO SEARCH::").strip().lower()
        df = self.results[["TeamName", "Position"]].dropna(subset="Position")
        filtered_df = df[df['TeamName'].str.strip().str.lower() == te_m]
        drivers = filtered_df.shape[0]
        avg_pos = round(filtered_df['Position'].mean(), 2)
        best_pos = int(filtered_df['Position'].min())
        summary = pd.DataFrame([{
            'TeamName': te_m.title(),
            'Drivers': drivers,
            'Avg_Position': avg_pos,
            'Best_Position': best_pos
        }])

        print(f"\n📊 Team Summary: {te_m.title()} 📊")
        print(tb.tabulate(summary, headers='keys', tablefmt='fancy_grid', showindex=False))

    def get_all_team_summary(self):
        df = self.results[['TeamName','Position']].dropna(subset='Position')
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        summary = df.groupby('TeamName').agg(
            Drivers=('Position', 'count'),
            Avg_Position=('Position', 'mean'),
            Best_Position=('Position', 'min')
        ).reset_index()

        print("\n📊 All Teams Summary 📊")
        print(tb.tabulate(summary, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    def best_driver_of_EACH_team(self):
        df = self.results[self.results['Position'].notna()]
        df = df[['TeamName', "FullName", 'Position']]
        df['Position'] = pd.to_numeric(df['Position'], errors="coerce")
        idx = df.groupby("TeamName")['Position'].idxmin()
        best_drivers = df.loc[idx].reset_index(drop=True)

        print("\n⭐ Best Driver of Each Team ⭐")
        print(tb.tabulate(best_drivers, headers='keys', tablefmt='fancy_grid', showindex=False))


class LapTimeAnalyzer():
    def __init__(self, year, gp_round, session_type):
        self.year = year
        self.round = gp_round
        self.session_type = session_type
        self.session = fastf1.get_session(year, gp_round, session_type)
        self.session.load(telemetry=True, messages=True)
        self.laps = self.session.laps

    def top_3_fastest_lap(self):
        valid_laps = self.laps[self.laps['LapTime'].notna()][['Driver','Team','LapTime']]
        fastest_laps = valid_laps.sort_values('LapTime').head(3).reset_index(drop=True)

        print("\n⚡ Top 3 Fastest Laps ⚡")
        print(tb.tabulate(fastest_laps, headers="keys", tablefmt='fancy_grid', showindex=False))

    def pit_stop_analysis(self):
        pitstops = self.laps[['Driver','Team','PitInTime','LapNumber']].dropna(subset='PitInTime')
        sorted_pitstops = pitstops.sort_values('LapNumber').reset_index(drop=True)

        print("\n🛑 Pit Stop Analysis 🛑")
        print(tb.tabulate(sorted_pitstops, headers="keys", tablefmt='fancy_grid', showindex=False))
    
    def stint_duration_team(self):
        tea_m = input("ENTER THE TEAM::").lower().strip()
        laps_df = self.laps[self.laps['Team'].str.lower().str.strip() == tea_m]
        stint_dur = laps_df.groupby(['Driver', 'Stint']).size().reset_index(name='StintDuration')
        pit_times = laps_df.groupby(['Driver', 'Stint'])['PitInTime'].min().reset_index()
        needed_df = stint_dur.merge(pit_times, on=['Driver','Stint'])
        needed_df = needed_df.sort_values('PitInTime').reset_index(drop=True).dropna(subset='PitInTime')

        print(f"\n⏱️ Stint Durations for {tea_m.title()} ⏱️")
        print(tb.tabulate(needed_df, headers="keys", tablefmt='fancy_grid', showindex=False))


class Statisticss:
    def __init__(self, year_, driver_name=None):
        self.year = year_
        self.driver_name = driver_name.lower().strip() if driver_name else None
        self.results_data = {}
    
    def _load_race_results(self, rnd):
        try:
            session = fastf1.get_session(self.year, rnd, 'R')
            session.load()
            results = session.results
            
            for _, row in results.iterrows():
                drv = row['FullName']
                team = row['TeamName']
                position = row['Position']
                points = row['Points']

                if self.driver_name and self.driver_name not in drv.lower():
                    continue

                if drv not in self.results_data:
                    self.results_data[drv] = {
                        'Team': team,
                        'Points': 0,
                        'Wins': 0,
                        'Podiums': 0,
                        'Races': 0
                    }

                self.results_data[drv]['Points'] += points
                self.results_data[drv]['Races'] += 1
                if position == 1:
                    self.results_data[drv]['Wins'] += 1
                if position <= 3:
                    self.results_data[drv]['Podiums'] += 1
                    
        except Exception as e:
            print(f"⚠️ Failed to load round {rnd}: {e}")

    def driver_overall_season(self):
        schedule = get_event_schedule(self.year)
        valid_rounds = schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]['RoundNumber'].tolist()
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(self._load_race_results, valid_rounds)
        df = pd.DataFrame.from_dict(self.results_data, orient='index')
        df.index.name = 'Driver'
        df.reset_index(inplace=True)
        df.sort_values(by='Points', ascending=False, inplace=True)

        print("\n📈 Driver Season Statistics 📈")
        print(tb.tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))


class Weather:
    def __init__(self, year, round_number, session_type):
        self.session = fastf1.get_session(year, round_number, session_type)
        self.session.load()

    def weathercond(self):
        circuit_info = self.session.get_circuit_info()
        raw_dict = vars(circuit_info)
        weather = raw_dict.pop("weather_data", {})
        combined = {**raw_dict, **weather}
        df = pd.DataFrame([combined])

        print("\n🌦️ Weather & Circuit Info 🌦️")
        print(tb.tabulate(df.values.tolist(), headers=df.columns.tolist(), tablefmt="fancy_grid", showindex=False))


# ✅ Call and test
race = f1teamsession(2024,"R",18)
race.indivdual_session_results()