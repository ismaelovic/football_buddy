import os
from datetime import datetime

from api_football import FootballDataAgent
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain.tools import Tool

load_dotenv()
# Initialize Gemini


class FootballDataCrew:
    """Football Data crew"""

    # Initialize Gemini using CrewAI's LLM
    api_key = os.getenv("GEMINI_API_KEY")
    gemini = LLM(model="gemini-2.0-flash-exp")

    # Create an instance of the API
    football_api = FootballDataAgent()

    competition_standings_tool = Tool(
        name="get_league_standings",
        func=football_api.get_league_standings,
        description="Get current league standings for a specific competition. Input should be a competition ID (e.g., 2021 for Premier League)",
    )
    team_performance_tool = Tool(
        name="get_team_performance",
        func=football_api.get_team_performance,
        description="Get current current team performance data for a given tgeam. Input should be a team ID (e.g., 61 for Chelsea FC)",
    )
    player_info_tool = Tool(
        name="get_player_info",
        func=football_api.get_player_info,
        description="Get detailed player information for a given team. Input should be a team ID (e.g., 61 for Chelsea FC)",
    )
    games_by_player_id_tool = Tool(
        name="get_games_by_player_id",
        func=football_api.get_games_by_player_id,
        description="Get detailed match information for matches a given player has played. Input should be a player ID (e.g., 102603 for Enzo Fernandez)",
    )

    # ID Parser Agent - Specialized in interpreting user requests
    id_parser_agent = Agent(
        role="Football ID Parser",
        goal="Accurately interpret user requests and map them to correct team and competition IDs.",
        backstory="""You are an expert in football database management with comprehensive 
        knowledge of team and competition IDs. Your specialty is interpreting natural 
        language requests and converting them into the correct database identifiers. 
        You have memorized all major team and competition IDs and can easily map 
        team names to their corresponding IDs that work in the Football Data API.

        You know all major competition IDs:
        - Premier League (England) has ID 2021
        - La Liga (Spain) has ID 2014
        - Bundesliga (Germany) has ID 2002
        - Serie A (Italy) has ID 2019
        - Ligue 1 (France) has ID 2015
        - Champions League has ID 2001
        - Europa League has ID 2146

        And you know the IDs of major teams:
        Premier League teams:
        - Chelsea has ID 61
        - Arsenal has ID 57
        - Manchester United has ID 66
        - Liverpool has ID 64
        - Manchester City has ID 65
        - Tottenham has ID 73

        Other major European teams:
        - Real Madrid has ID 86
        - Barcelona has ID 81
        - Bayern Munich has ID 5
        - Juventus has ID 109
        - Inter Milan has ID 108
        - PSG has ID 524

        When someone mentions a team or competition by name, you automatically know 
        the correct ID to use in the API calls.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini,
    )

    # Data Fetcher Agent - Specialized in using the tools to get data
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_fetcher_agent = Agent(
        role="Football Data Fetcher",
        goal=f"To retrieve the relevant football data requested by the user as of {current_timestamp}",
        backstory=f"""You are a focused data retrieval specialist who knows what tools to utilize for the given need.
    You only retrieve data that directly answers the user's query, nothing more, nothing less.""",
        verbose=True,
        allow_delegation=False,
        tools=[
            competition_standings_tool,
            team_performance_tool,
            player_info_tool,
            games_by_player_id_tool,
        ],
        llm=gemini,
    )

    reporter_agent = Agent(
        role="Focused Football Data Reporter",
        goal="Create fun, engaging and targeted summaries that directly address the user's specific query",
        backstory="""You are a precise football enthusiast who focuses solely on answering 
    the exact question asked by the user. You never stray from the specific topic requested.

    Your reporting principles:
    1. Address ONLY what was specifically asked
    2. Your answers are ALWAYS based on data provided from the data fetcher agent
    3. Don't speculate beyond the provided data
    4. If something specific was asked but data isn't available, you clearly state that. You don't need to answer a question that can't be answered.

    You maintain focus by:
    - Reading the user's question carefully
    - Reporting only relevant data points
    - Avoiding tangential information
    - Staying within the scope of the question
    - Being concise and direct""",
        verbose=True,
        allow_delegation=False,
        llm=gemini,
    )

    parse_request = Task(
        description="""Analyze this request: {topic}
        Identify the team and competition IDs. If no competition is mentioned then focus on the teams domestic competition. Fx For Real Madrid that would be La Liga. For Bayern Munchen it would be Bundesliga""",
        expected_output="ONLY a JSON with 'team_id' and 'competition_id', please dont include any text or markdown json´´´ of any sort before or after the output.",
        agent=id_parser_agent,
    )

    gather_data = Task(
        description="""Using the IDs provided by the parser, fetch relevant data that matches the user's request.

    Analyze the parsed request to determine:
    1. What specific information the user is asking for
    2. Which IDs are needed for those specific data points
    3. Which tools are relevant for this particular request

    Then:
    - Only fetch data that's relevant to the user's query
    - Only use the tools that are needed for this specific request
    - Skip any data collection that isn't relevant to the user's question


    Focus on efficiency and relevance rather than collecting all possible data.""",
        expected_output="A focused collection of data that directly addresses the user's specific request, containing only the relevant information asked for.",
        agent=data_fetcher_agent,
        context=[parse_request],
    )

    create_summary = Task(
        description="""FOCUS ONLY ON THIS EXACT QUESTION: '{topic}'

    STRICT RESPONSE RULES:
    1. Read the user's question carefully and identify EXACTLY what is being asked
    2. Only use data that DIRECTLY answers this specific question
    3. IGNORE all other data points, even if interesting or related
    4. If no relevant data exists to answer THIS SPECIFIC question, respond: "I don't have enough specific data to answer this question about [exact topic asked]."

    EXAMPLE:
    - If asked "Has Player X been good?" → Only use data about Player X's performance
    - If asked "How many goals in the Premier League?" → Only provide goal statistics
    - If asked about one team → Ignore all other team's data

    DO NOT:
    - Include any data that doesn't directly answer the specific question
    - Make broader conclusions from the available data
    - Add context beyond what was specifically asked
    - Mention any statistics or facts not directly related to the question""",
        expected_output="A response that ONLY answers the specific question asked, nothing more.",
        agent=reporter_agent,
        context=[gather_data],
    )

    def crew(self) -> Crew:
        """Creates the Football Buddy crew"""
        return Crew(
            agents=[self.id_parser_agent, self.data_fetcher_agent, self.reporter_agent],
            tasks=[self.parse_request, self.gather_data, self.create_summary],
            process=Process.sequential,
            verbose=True,
        )