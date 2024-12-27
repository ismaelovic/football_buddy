import os
from typing import ClassVar, Dict

import requests
from dotenv import load_dotenv

load_dotenv()


class FootballDataAgent:
    name: str = "get_league_standings"  # Changed to be more specific to the action
    description: str = (
        "Get current league standings for a specific competition. Input should be a competition ID (e.g., 2021 for Premier League)"
    )
    api_key: str = str(os.getenv("FOOTBALL_DATA_API_KEY"))
    base_url: ClassVar[str] = "http://api.football-data.org/v4"
    headers: ClassVar[Dict[str, str]] = {
        "X-Auth-Token": api_key,
        "X-Unfold-Goals": "true",
    }

    def get_league_standings(self, competition_id) -> dict:
        """Get current league standings for a specific competition.
        Args:
        competition_id (int): The ID of the competition (e.g., 2021 for Premier League)

        Returns:
        dict: Current standings for the specified competition
        """
        url = f"{self.base_url}/competitions/{competition_id}/standings"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_team_performance(self, team_id, competition_id=None):
        """Get team's matches across all or specific competition.
        Args:
            team_id (int): The ID of the team (e.g., 61 for Chelsea)
            competition_id (int, optional): The ID of the competition to filter matches

        Returns:
            dict: Match history for the specified team
        """
        url = f"{self.base_url}/teams/{team_id}/matches"
        params = {}
        if competition_id:
            params["competition"] = competition_id

        response = requests.get(url, headers=self.headers, params=params)
        return response.json()

    def get_player_info(self, team_id):
        """Get squad and player information for a team"""
        url = f"{self.base_url}/teams/{team_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_games_by_player_id(self, player_id):
        "http://api.football-data.org/v4"
        url = f"/persons/{player_id}/matches"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_head_to_head(self, team1_id, team2_id, limit=10):
        """Get head to head matches between two teams"""
        url = f"{self.base_url}/teams/{team1_id}/matches"
        params = {"limit": limit, "status": "FINISHED"}

        response = requests.get(url, headers=self.headers, params=params)
        matches = response.json()["matches"]

        h2h_matches = [
            match
            for match in matches
            if (
                match["homeTeam"]["id"] == team2_id
                or match["awayTeam"]["id"] == team2_id
            )
        ]

        return {"matches": h2h_matches}


if __name__ == "__main__":
    fb = FootballDataAgent()
    result = fb.get_team_performance("61", "2001")
    print(result)