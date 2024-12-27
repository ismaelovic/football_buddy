from crew import FootballDataCrew
from dotenv import load_dotenv

load_dotenv()


def run(user_input):
    "Run the crew."
    inputs = {"topic": user_input}

    return FootballDataCrew().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    print("Hi dude, Im your Football Assistant for today!")
    print("What would you like to know? Type 'quit' to kill me.")

    while True:
        user_input = input("\nYour question: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if user_input.strip():  # Check if input is not empty
            result = run(user_input)
            print("\nResponse:", result)
        else:
            print("Please enter a valid question.")