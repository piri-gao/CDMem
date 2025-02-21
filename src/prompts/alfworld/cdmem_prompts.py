import re

class CDMemPromptBuilder:
    def __init__(self):
        pass
    
    def get_inference_prompts(self, init_ob, fewshots, local_memories, short_memories, known_obs_history, action_guidance_history):        
        query = f"""
Role: As an expert in indoor navigation and manipulation, you can efficiently encode, memorize, and retrieve experiences based on action trajectories of exploration and manipulation. Thus you can rapidly adapt to new environments and efficiently complete tasks. 

Instruction: Given the environment, task, functions of containers(such as drawer, shelf, sinkbasin, fridge), locations of items(such as mug, lettuce, bread, alarm clock), action guidance and reflections from past trials, you need to interact with the environment to solve the task.

Exemplars: There are two exemplars to help you better understand how to interact with the environment and to solve the task.

{fewshots}

Goal Task: Now, based on the task background, task instruction, reference exemplars, functions of containers, locations of items and past reflections to output the correct action.
"""
        if known_obs_history:
            query += f"\nFunctions of Containers: {known_obs_history}"
        if action_guidance_history:
            query += f"\nAction Guidance:\n{action_guidance_history}"
        if len(local_memories) > 0:
            query += "\nPast Reflections:"
            for i, m in enumerate(local_memories):
                query += f'\nTrial {i}:\n{m.strip()}'
        query += f"""

Here is the task:
{init_ob}{short_memories}
"""
        return query 
    
    def get_expert_prompts(self, history_log, fewshots): 
        scenario = history_log.split("Here is the task:")[-1].strip()
        query = f"""
Role: As an expert in indoor navigation and manipulation, you can efficiently encode, memorize, and retrieve experiences based on action trajectories of exploration and manipulation. Thus you can rapidly adapt to new environments and efficiently complete tasks. 

Instruction: Given the environment, task, and action trajectories, you need to use an information chunking strategy (the process of reassembling various fragments into meaningful logical components, which aims to learn and remember complex information effectively) to efficiently encode this information from the environment observation (Expert Observations) and the action components (Expert Actions) as follows:
1. Expert Observations: Based on the information chunking strategy, you need to understand and summarize the environment and observations from two perspectives: (1) the location where items (such as mug, lettuce, bread, alarm clock) are placed, for example, drawer 1 has a mug, shelf 2 has an alarm clock;(2) the functions of some containers (such as drawer, shelf, sinkbasin, fridge). For example, you can clean lettuce with sinkbasin,  you can cool a mug with fridge. If no container's function needs to be summarized, output "None".
2. Expert Actions: Based on the information chunking strategy, you need to understand and summarize the action trajectories following the original execution order and ignore the thought process inside. If there are adjacent actions of the same type, some simplification can be made. 

Exemplars: There are three exemplars to help you better understand the information chunking strategy to complete the expert encoding task.

{fewshots}

Goal Task: Now, based on the task background, task instruction, and reference exemplars, you need to complete the task and give the expert encoding result, including expert observations and expert actions.

*** Input **** 

{scenario}

*** Expert Encoding Result **** 

Expert Observations:
Expert Actions: 
        """ 
        return query
                
    def get_reflection_prompts(self, history_log, is_success, fewshots, local_memories, expert_result):
        locations, functions, expert_actions = self._parser_expert_result(expert_result)
        expert_observations = f'''(1){locations}(2){functions}'''
        scenario = history_log.split("Here is the task:")[-1].strip()
        query: str = f"""
Role: As an expert in indoor navigation and manipulation, you can efficiently encode, memorize, and retrieve experiences based on action trajectories of exploration and manipulation and environmental observations. Thus you can rapidly adapt to new environments and efficiently complete tasks.

Instruction: {"You have successfully completed the task. Given the environment, task, action trajectories, and expert actions(summary of action trajectories), you need to reflect on the key actions that are critical to completing the task, which means that eliminating any of these actions would affect the completion of the task. " 
if is_success else
'''You were unsuccessful in completing the task. Given the environment, task, action trajectories, expert observations(location of items and functions of containers. items refer to mug, lettuce, bread, alarm clock, etc; containers refer to drawer, shelf, sinkbasin, fridge, etc), expert actions(summary of action trajectories), and past reflections(reflections you made in past trials) you must first consider what types of failure you meet and output corresponding reflections.
There are three types of failure:
Planning Failure: The task planning have issues, such as missing steps or misunderstandings.  Output the reflection of current planning issues and the correct plan. 
Search Failure: Continuously searching for an item but unable to find it. Output the item's location already searched and the reflection of the future search plan. For example, if you tried A and B but forgot C, devise a plan to achieve C with environment-specific actions. 
Operation Failure: The expected feedback was not received after performing the action, such as returning with "nothing happens," which means the current observation doesn't match the current action. For example, attempting to take something from cabinet 1 while at the location of cabinet 4 and then returning "nothing happens." Output the reflection of the failed reason and correct actions.
'''
}

Exemplars: There are {"two" if is_success else "three"} exemplars to help you better understand the reflection you should make.

{fewshots}

"""
        query += f'''
Goal Task: Now, based on the task background, task instruction, reference exemplars, and past reflections you need to complete the task and give your new reflection:

*** Input ***

{scenario}

'''
        if is_success:
            query += f"""
Expert Actions: {expert_actions}

"""
        else:
            query += f"""
Expert Actions: {expert_actions}
Expert Observations:{expert_observations}
"""
        if len(local_memories) > 0:
            query += 'Past Reflections:\n'
            for i, m in enumerate(local_memories):
                query += f'Trial #{i}: {m}\n'

        query +=f"""
*** Reflection Result***
Your reflection here, please start with: Reflection:
"""
        return query
    
    def env_summary_prompts(self, known_obs_history, env_fewshots):
        known_obs = known_obs_history['known_obs']
        increment_known_obs = known_obs_history['increment_known_obs']
        query = f"""
Role: As an expert in indoor navigation and manipulation, you can efficiently encode, memorize, and retrieve experiences based on action trajectories of exploration and manipulation and environmental observations. Thus you can rapidly adapt to new environments and efficiently complete tasks. 

Instruction: Given multiple experiences of expert observations(functions of containers, containers refer to drawer, shelf, sinkbasin, fridge, etc) and environmental summary you made in past trials, you need to summarize them in an new environmental summary. For example, given two expert observations, "I can clean mug with sinkbasin" and "I can clean egg with sinkbasin", and your past summary "I can cool items with fridge" , a new summary can be "I can clean items with sinkbasin, I can cool items with fridge".

Exemplars: There are two exemplars to help you better understand the summary you should make.

{env_fewshots}

"""
        
        query += f"""Goal Task: Now, based on the task background, task instruction, reference exemplars, and past summary, you need to complete the task and give your new summary:"""
        query += f"""
        
*** Input ***

"""
        for i, m in enumerate(increment_known_obs):
            query += f'Expert Observation #{i}: {m}\n'
        if known_obs:
                    query += f"""Past Summary:{known_obs}"""
        query += f"""
        
*** Summary Result***

"""
        return query
        
    def task_summary_prompts(self, action_guidance_history, task_fewshots, is_success):
        action_guidance = action_guidance_history['action_guidance']
        increment_action_guidance = action_guidance_history['increment_action_guidance']
        query = f"""Role: As an expert in indoor navigation and manipulation, you can efficiently encode, memorize, and retrieve experiences based on action trajectories of exploration and manipulation and environmental observations. Thus you can rapidly adapt to new environments and efficiently complete tasks.\n\n"""
        if is_success:
            query += f"""Instruction: Given multiple experiences of task names, corresponding reflections(key actions which are critical to completing the task) and summaries you made in past trials,  you need to summarize these experiences and past summaries as a task summary containing the general planning such as "I should first find the item, then heat it with microwave, and put it in/on container at last.").\n\n"""
        else:
            query += f"""Instruction: Given multiple experiences containing task names, corresponding reflections(type of failure, description of the failure situation, and a future plan)  and summaries you made in past trials,  you need to summarize these experiences and past summaries as a task summary containing the failure situations and corresponding future plans.\n\n"""
        query += f"""Exemplars: There is an exemplar to help you better understand the summary you should make.

{task_fewshots}

Goal Task: Now, based on the task background, task instruction, reference exemplars, and past summaries, you need to complete the task and give your new summary:

*** Input ***
        """
        for i, m in enumerate(increment_action_guidance):
            task = m['task']
            reflection = m['reflection']
            if is_success:
                experience = f"""Task Name: {task}\Key Action: {reflection}"""
            else:
                experience = f"""Task Name: {task}\nReflection: {reflection}"""
            query += f"""
Experience #{i}:\n{experience}\n"""
        if action_guidance:
            query += f"""Past Summary:\n{action_guidance}"""
        query += f"""
        
***Summary Result***
""" 
        return query
    
    def _parser_expert_result(self, expert_result):
        location = function = action = ''
        location_match = re.search(r'\(1\)locations:(.*?)\(2\)functions:', expert_result, re.DOTALL)
        function_match = re.search(r'\(2\)functions:(.*?)Expert Actions:', expert_result, re.DOTALL)
        action_match = re.search(r'Expert Actions:(.*)', expert_result, re.DOTALL)
        if location_match:
            location = location_match.group(1).strip() 
        if function_match:
            function = function_match.group(1).strip() 
        if action_match:
            action = action_match.group(1).strip()
        return location, function, action
        