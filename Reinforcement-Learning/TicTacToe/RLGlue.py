import numpy as np

class RLGlue:
    """RLGlue class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_class, agent1_class, agent2_class):
        self.environment = env_class()

        self.agent1 = {"agent": agent1_class(), "curr_reward": None, "total_reward": None}
        self.agent2 = {"agent": agent2_class(), "curr_reward": None, "total_reward": None}
        self.agents = {-1: self.agent1, 1: self.agent2}
        self.current_agent = 0

        self.num_steps = None
        self.num_episodes = None

    def rl_init(self, agent1_init_info={}, agent2_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agents[-1]["agent"].agent_init(agent1_init_info)
        self.agents[1]["agent"].agent_init(agent2_init_info)
        self.agents[-1]["curr_reward"] = 0
        self.agents[1]["curr_reward"] = 0
        self.agents[-1]["total_reward"] = 0
        self.agents[1]["total_reward"] = 0
        self.num_steps = 0
        self.num_episodes = 0

    def switch_players(self):
        self.current_agent *= -1

    def rl_start(self, agent_start_info={}, env_start_info={}):      
        self.agents[-1]["curr_reward"] = 0
        self.agents[1]["curr_reward"] = 0
        self.agents[-1]["total_reward"] = 0
        self.agents[1]["total_reward"] = 0
        self.num_steps = 0

        # who starts?
        if np.random.rand() < 0.5:
            self.current_agent = -1
            self.start = -1
        else:
            self.current_agent = 1
            self.start = 1

        # print(f"start {self.start}")

        state, mask = self.environment.env_start()

        # 1st player takes first move
        state, terminal, mask = self.player_start(state, mask)
        self.num_steps += 1
        # 2nd player takes first move
        state, terminal, mask = self.player_start(state, mask)
        self.num_steps += 1
        
        self.state = state
        self.mask = mask
        observation = (state, mask, terminal)
        return observation

    def player_start(self, state, mask):
        action = self.agents[self.current_agent]["agent"].agent_start(state, mask)
        # print(f"player {self.current_agent} chose {action}")
        reward, state, terminal, mask = self.environment.env_step(self.current_agent, action)
        self.agents[self.current_agent]["curr_reward"] = reward
        self.agents[self.current_agent]["total_reward"] += reward
        self.switch_players()
        return state, terminal, mask

    def rl_step(self):
        # 1st player step
        reward, state, terminal, mask = self.player_step(self.state, self.mask)
        if terminal:
            self.finish_game(reward)
            return True
        self.switch_players()
        
        # 2nd player step
        reward, state, terminal, mask = self.player_step(state, mask)
        if terminal:
            self.finish_game(reward)
            return True
        self.switch_players()

        self.mask = mask

        return False

    def finish_game(self, reward):
        self.num_episodes += 1
        self.agents[self.current_agent]["agent"].agent_end(reward)
        self.switch_players()
        self.agents[self.current_agent]["agent"].agent_end(-reward)
        self.agents[self.current_agent]["total_reward"] -= reward


    def player_step(self, state, mask):
        reward = self.agents[self.current_agent]["curr_reward"]
        action = self.agents[self.current_agent]["agent"].agent_step(reward, state, mask)
        # print(f"player {self.current_agent} chose {action}")
        reward, state, terminal, mask = self.environment.env_step(self.current_agent, action)
        self.agents[self.current_agent]["curr_reward"] = reward
        self.agents[self.current_agent]["total_reward"] += reward
        self.num_steps += 1
        self.state = state
        return reward, state, terminal, mask

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            is_terminal = self.rl_step()
        # print("game over")
        return is_terminal

    def rl_return(self):
        return (self.agent1["total_reward"], self.agent2["total_reward"]), self.start, self.state

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
