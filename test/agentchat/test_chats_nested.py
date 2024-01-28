from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST
import pytest
from conftest import skip_openai
import autogen

try:
    import openai
except ImportError:
    skip = True
else:
    skip = False or skip_openai


@pytest.mark.skipif(skip, reason="openai not installed OR requested to skip")
def test_chats_nested():
    config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    llm_config = {"config_list": config_list}

    financial_tasks = [
        """On which days in 2024 was Microsoft Stock higher than $370? Put results in a table and don't use ``` ``` to include table.""",
        """Investigate the possible reasons of the stock performance.""",
    ]

    assistant = autogen.AssistantAgent(
        "Inner-assistant",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    code_interpreter = autogen.UserProxyAgent(
        "Inner-code-interpreter",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
        default_auto_reply="",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    groupchat = autogen.GroupChat(
        agents=[assistant, code_interpreter],
        messages=[],
        speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
        allow_repeat_speaker=False,
        max_round=8,
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=llm_config,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
    )

    financial_assistant_1 = autogen.AssistantAgent(
        name="Financial_assistant_1",
        llm_config={"config_list": config_list},
    )

    user = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        code_execution_config={
            "last_n_messages": 1,
            "work_dir": "tasks",
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    # def chat(inner_monologue_agent, recipient, messages, sender, config):
    #     return True, recipient.initiate_chat(inner_monologue_agent, message=messages[0], takeaway_method="llm")

    financial_assistant_1.register_reply(
        [autogen.Agent, None], autogen.nested_chat_reply(manager, takeaway_method="llm"), 3
    )

    user.initiate_chat(financial_assistant_1, message=financial_tasks[0])


if __name__ == "__main__":
    test_chats_nested()
