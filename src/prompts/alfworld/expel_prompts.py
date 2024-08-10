class ExpelPromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories, guidelines):
        query = 'Interact with a household to solve a task. You may take maximum of 20 steps. Here are two examples.\n'
        query += fewshots
        if len(local_memories) > 0:
            query += '\n\nYour memory for the task below:'
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:{m.strip()}'
        # if guidelines and guidelines != 'None':
        #     query += f'\nThe following are some experience you gather on a similar task of completing a household task by interacting in a household environment. Use these as references to help you perform this task:\n{guidelines}\n'
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

        query += '\nYour Job is to generate your new plan in one concise sentence. Directly output the content of the plan without including the word "plan."'
        return query

    def get_pair_guidelines_prompts(self, fail_traj, success_traj, existing_rules):
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. Here are the two previous trials to compare and critique:

SUCCESSFUL TRIAL:
{success_traj}

FAILED TRIAL:
{fail_traj}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, Your job is to generate a new list of rules. The new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. 
Here are some example rules:
1. When searching for an item, consider the nature of the item and its typical usage. For example, a pan is more likely to be found on a stoveburner or countertop rather than in a cabinet or drawer
2. lf an attempt to interact with an item fails or does not progress the task, reassess the situation and consider alternative actions or locations before repeating the same action
3. Always confrm the presence of an item before attempting to interact with it.
Note that the new list of rules contains up to 5 rules and Do Not put a blank line between two rules. Now it's your Turn.  
'''
        return query

    def get_success_guidelines_prompts(self, success_trajs, existing_rules):
        query = f'''
You are a housekeeper robot. The agent was placed in a household environment and a task to complete. 
Here are the successful trials:
{success_trajs}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, Your job is to generate a new list of rules which are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. 
Here are some example rules: 
1. When searching for an item, consider the nature of the item and its typical usage. For example, a pan is more likely to be found on a stoveburner or countertop rather than in a cabinet or drawer
2. lf an attempt to interact with an item fails or does not progress the task, reassess the situation and consider alternative actions or locations before repeating the same action
3. Always confrm the presence of an item before attempting to interact with it.
Note that the new list of rules contains up to 5 rules and Do Not put a blank line between two rules. Now it's your Turn.  
'''
        return query