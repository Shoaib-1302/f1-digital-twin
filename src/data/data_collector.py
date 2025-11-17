"""
Data Collection Module
Handles data fetching from Ergast API, OpenF1 API, and news sources
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

class ErgastAPIClient:
    """Client for Ergast Developer API (Historical F1 Data)"""
    
    BASE_URL = "https://ergast.com/api/f1"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/raw/ergast")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_seasons(self, start_year: int = 1950, end_year: int = 2024) -> List[int]:
        """Get list of F1 seasons"""
        url = f"{self.BASE_URL}/seasons.json"
        response = requests.get(url, params={"limit": 100})
        data = response.json()
        seasons = [int(s['season']) for s in data['MRData']['SeasonTable']['Seasons']]
        return [s for s in seasons if start_year <= s <= end_year]
    
    def get_race_results(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Get race results for a season or specific round"""
        cache_file = self.cache_dir / f"results_{year}_{round_num or 'all'}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
        else:
            url = f"{self.BASE_URL}/{year}"
            if round_num:
                url += f"/{round_num}"
            url += "/results.json"
            
            response = requests.get(url, params={"limit": 1000})
            data = response.json()
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            time.sleep(0.5)  # Rate limiting
        
        races = data['MRData']['RaceTable']['Races']
        results = []
        
        for race in races:
            race_info = {
                'season': race['season'],
                'round': race['round'],
                'race_name': race['raceName'],
                'circuit': race['Circuit']['circuitName'],
                'date': race['date'],
                'country': race['Circuit']['Location']['country']
            }
            
            for result in race['Results']:
                row = race_info.copy()
                row.update({
                    'position': result.get('position'),
                    'driver_id': result['Driver']['driverId'],
                    'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    'constructor_id': result['Constructor']['constructorId'],
                    'constructor_name': result['Constructor']['name'],
                    'grid': result.get('grid'),
                    'laps': result.get('laps'),
                    'status': result.get('status'),
                    'points': float(result.get('points', 0)),
                    'fastest_lap_rank': result.get('FastestLap', {}).get('rank'),
                    'fastest_lap_time': result.get('FastestLap', {}).get('Time', {}).get('time')
                })
                results.append(row)
        
        return pd.DataFrame(results)
    
    def get_qualifying_results(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Get qualifying results"""
        url = f"{self.BASE_URL}/{year}"
        if round_num:
            url += f"/{round_num}"
        url += "/qualifying.json"
        
        response = requests.get(url, params={"limit": 1000})
        data = response.json()
        
        races = data['MRData']['RaceTable']['Races']
        results = []
        
        for race in races:
            for qual in race['QualifyingResults']:
                results.append({
                    'season': race['season'],
                    'round': race['round'],
                    'race_name': race['raceName'],
                    'driver_id': qual['Driver']['driverId'],
                    'driver_name': f"{qual['Driver']['givenName']} {qual['Driver']['familyName']}",
                    'constructor_id': qual['Constructor']['constructorId'],
                    'position': qual['position'],
                    'q1': qual.get('Q1'),
                    'q2': qual.get('Q2'),
                    'q3': qual.get('Q3')
                })
        
        time.sleep(0.5)
        return pd.DataFrame(results)
    
    def get_driver_standings(self, year: int) -> pd.DataFrame:
        """Get final driver standings for a season"""
        url = f"{self.BASE_URL}/{year}/driverStandings.json"
        response = requests.get(url)
        data = response.json()
        
        standings_list = data['MRData']['StandingsTable']['StandingsLists'][0]
        standings = []
        
        for driver in standings_list['DriverStandings']:
            standings.append({
                'season': year,
                'position': driver['position'],
                'driver_id': driver['Driver']['driverId'],
                'driver_name': f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}",
                'constructor_id': driver['Constructors'][0]['constructorId'],
                'points': float(driver['points']),
                'wins': int(driver['wins'])
            })
        
        time.sleep(0.5)
        return pd.DataFrame(standings)
    
    def get_lap_times(self, year: int, round_num: int, lap_num: Optional[int] = None) -> pd.DataFrame:
        """Get lap times for a specific race"""
        url = f"{self.BASE_URL}/{year}/{round_num}/laps"
        if lap_num:
            url += f"/{lap_num}"
        url += ".json"
        
        response = requests.get(url, params={"limit": 2000})
        data = response.json()
        
        races = data['MRData']['RaceTable']['Races']
        if not races:
            return pd.DataFrame()
        
        laps = races[0]['Laps']
        results = []
        
        for lap in laps:
            lap_num = lap['number']
            for timing in lap['Timings']:
                results.append({
                    'season': year,
                    'round': round_num,
                    'lap': lap_num,
                    'driver_id': timing['driverId'],
                    'position': timing['position'],
                    'time': timing['time']
                })
        
        time.sleep(0.5)
        return pd.DataFrame(results)


class OpenF1APIClient:
    """Client for OpenF1 API (Real-time and Recent Data)"""
    
    BASE_URL = "https://api.openf1.org/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_sessions(self, year: Optional[int] = None, session_type: Optional[str] = None) -> pd.DataFrame:
        """Get session information"""
        params = {}
        if year:
            params['year'] = year
        if session_type:
            params['session_type'] = session_type
        
        response = self.session.get(f"{self.BASE_URL}/sessions", params=params)
        data = response.json()
        return pd.DataFrame(data)
    
    def get_lap_times_openf1(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """Get lap times from OpenF1"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
        
        response = self.session.get(f"{self.BASE_URL}/laps", params=params)
        data = response.json()
        return pd.DataFrame(data)
    
    def get_car_data(self, session_key: int, driver_number: int) -> pd.DataFrame:
        """Get car telemetry data"""
        params = {
            'session_key': session_key,
            'driver_number': driver_number
        }
        
        response = self.session.get(f"{self.BASE_URL}/car_data", params=params)
        data = response.json()
        return pd.DataFrame(data)
    
    def get_position_data(self, session_key: int) -> pd.DataFrame:
        """Get position tracking data"""
        params = {'session_key': session_key}
        response = self.session.get(f"{self.BASE_URL}/position", params=params)
        data = response.json()
        return pd.DataFrame(data)


class NewsAPIClient:
    """Client for collecting F1 news articles"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def get_f1_news(self, days_back: int = 30, language: str = 'en') -> pd.DataFrame:
        """Fetch recent F1 news articles"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': 'Formula 1 OR F1',
            'from': from_date,
            'language': language,
            'sortBy': 'publishedAt',
            'apiKey': self.api_key
        }
        
        response = requests.get(f"{self.base_url}/everything", params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                })
            
            return pd.DataFrame(articles)
        else:
            print(f"Error fetching news: {response.status_code}")
            return pd.DataFrame()


class F1DataCollector:
    """Main data collector orchestrating all sources"""
    
    def __init__(self, news_api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.ergast = ErgastAPIClient(cache_dir)
        self.openf1 = OpenF1APIClient()
        self.news = NewsAPIClient(news_api_key) if news_api_key else None
    
    def collect_season_data(self, year: int) -> Dict[str, pd.DataFrame]:
        """Collect all data for a complete season"""
        print(f"Collecting data for {year} season...")
        
        data = {
            'results': self.ergast.get_race_results(year),
            'qualifying': self.ergast.get_qualifying_results(year),
            'standings': self.ergast.get_driver_standings(year)
        }
        
        print(f"Collected {len(data['results'])} race results")
        print(f"Collected {len(data['qualifying'])} qualifying results")
        print(f"Collected {len(data['standings'])} driver standings")
        
        return data
    
    def collect_multi_season_data(self, start_year: int, end_year: int) -> Dict[str, pd.DataFrame]:
        """Collect data across multiple seasons"""
        all_results = []
        all_qualifying = []
        all_standings = []
        
        for year in range(start_year, end_year + 1):
            print(f"Processing {year}...")
            season_data = self.collect_season_data(year)
            
            all_results.append(season_data['results'])
            all_qualifying.append(season_data['qualifying'])
            all_standings.append(season_data['standings'])
        
        return {
            'results': pd.concat(all_results, ignore_index=True),
            'qualifying': pd.concat(all_qualifying, ignore_index=True),
            'standings': pd.concat(all_standings, ignore_index=True)
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Save collected data to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            filepath = output_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")


if __name__ == "__main__":
    # Example usage
    collector = F1DataCollector()
    
    # Collect recent seasons
    data = collector.collect_multi_season_data(2020, 2024)
    
    # Save to disk
    collector.save_data(data, Path("data/raw"))
