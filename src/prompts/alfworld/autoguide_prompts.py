class AutoguidePromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories, guidelines):
        query = 'Interact with a household to solve a task. You may take maximum of 20 steps. Here are two examples.\n'
        query += fewshots
        if len(local_memories) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:\n{m.strip()}'
        if len(guidelines) > 0:
            query += '\nThe following are some experience you gather on a similar task of completing a household task by interacting in a household environment. Use these as references to help you perform this task:'
            for i, m in enumerate(guidelines):
                query += f'\n{i}.{m.strip()}'
        query += f"\nHere is the task:\n{init_ob}"
        query += short_memories
        return query
                
    def get_reflection_prompts(self, log_str, fewshots, local_memories):
        scenario = log_str.split("Here is the task:")[-1].strip()
        query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{fewshots}

{scenario}"""

        if len(local_memories) > 0:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(local_memories):
                query += f'Trial #{i}: {m}\n'

        query += '\n\nNew plan:'
        return query

    def get_pair_guidelines_prompts(self, fail_traj, success_traj):
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. You will be provided with a failed and a successful trajectory of the same task. What is the first action that differs between the two trajectories? Why do you think it makes one trajectory failed and the other successful? Based on your answer, generate an action guideline to make future task avoid the same mistake. The guideline should specify what to do in what situation in the format of "When in what status,  you should (or should not)... ". For the 'When in what status' part, directly use the words in SUMMARIZATION. Here are two examples:Example 1: When looking for an object, if you want to find a kitchen-related object like a spatula, you should start from the most possible locations.Example 2: When looking for an object and found the desired object at the location, You should only take the exact object that you want.Strictly follow what the successful trajectory does and never suggest actions that the successful trajectory didn't do. When referring to actions, use the allowed action format. You should make your answer concise, limit your answer within 128 tokens,and put your answer in the format: 'Reasoning: ...Guideline: ...'.

Failed Trajectory: {fail_traj}
Successful Trajectory: {success_traj}
'''
        return query

    def get_guideline_selection_prompts(self, guideline_list, init_ob, short_memories):
        if len(guideline_list) > 0:
            guidelines = ''
            for i, m in enumerate(guideline_list):
                guidelines += f'\n{i}.{m.strip()}'
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. You will be equipped with the following resources:
1. A list of action guidelines with valuable guidelines.
2. Trajectory history, which includes recent observations and actions.
Not all guidelines are useful to generate the next action. Please select the guidelines that are useful and relevant to the nextaction given the trajectory and recent observations. To generate the next action, which guidelines from the provided guidelinesare most useful to directly tell you what to do for the next action? You can select up to 2 guidelines, and put the indices of theselected guidelines in a python list. For example if you select guideline 1, 5, answer: [1, 5]. If none of them are useful forgenerating the next action, answer the empty list [].

Guidelines:
{guidelines}
Trajectory history:
{init_ob}
{short_memories}
'''
        return query

    def get_success_guidelines_prompts(self, success_trajs, existing_rules):
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. 
Here are the successful trials:
{success_trajs}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, generate a new list of rules which are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future.      
'''
        return query

    def get_status_summary_prompts(self, fewshots, init_ob, short_memories):
        query = f'''
You'll get a snippet of a trajectory of an text-based ALFRED task. Your job is to generate a brief and general summarization ofthe current status following 'SUMMARIZATION: '.Keep it broad and general, avoid any information about specific objects and locations.  
Here is an example:
{fewshots}
Now it's your turn:
{init_ob}
{short_memories}
'''
        return query

    def get_status_matching_prompts(self, status_list, new_status):
        if len(status_list) > 0:
            seen_summaries = ''
            for i, m in enumerate(status_list):
                seen_summaries += f'\n{i}#:\n{m.strip()}'
        else:
            seen_summaries = 'None'
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. A task trajectory can be long. Therefore the assistant summarizes the status of each step.For different task with the same status, the summarization should be the same, therefore please ignore any information about instructions or products.You will be provided with the following:
1. A list of summarizations the assistant saw in the past.
2. A newly generated summarization.
Please determine if any summarization from the list matches the exact same status as the newly generated one. If yes, answerthe index of the corresponding summarization, for example "Answer: 2"; otherwise, "Answer: None".

Seen Summarizations:{seen_summaries}
Newly Generated Summarization:{new_status}
'''
        return query