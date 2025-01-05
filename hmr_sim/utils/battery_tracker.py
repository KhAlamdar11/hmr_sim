import csv

class BatteryTracker:
    def __init__(self):
        # Initialize a dictionary to track batteries over time
        self.battery_data = {}  # {agent_id: [battery_levels_over_time]}
        self.time_step = 0  # Track simulation time steps

    def update(self, agents):
        """
        Update the battery tracker with the current state of agents.
        Args:
            agents (list): List of Agent objects.
        """
        print(len(agents))
        for agent in agents:
            if agent.type == 1 or agent.type == "UAV":
                if agent.agent_id not in self.battery_data:
                    # Initialize battery list for new agents
                    self.battery_data[agent.agent_id] = []
                # Append the current battery level
                self.battery_data[agent.agent_id].append((self.time_step, agent.battery))

        # Increment time step
        self.time_step += 1

    def save_to_file(self, file_name="battery_data.csv"):
        """
        Save the battery data to a CSV file.
        Args:
            file_name (str): Name of the file to save the data.
        """
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["agent_id", "time_step", "battery"])
            # Write battery data
            for agent_id, battery_history in self.battery_data.items():
                for time_step, battery in battery_history:
                    writer.writerow([agent_id, time_step, battery])
        print(f"Battery data saved to {file_name}")
