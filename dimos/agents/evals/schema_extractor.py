# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract skill schemas from blueprints and skill containers."""

from dimos.agents.spec import ToolSchemaList
from dimos.core.blueprints import ModuleBlueprintSet
from dimos.protocol.skill.skill import SkillContainer
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def extract_skills_from_container(
    container: SkillContainer,
) -> ToolSchemaList:
    """Extract OpenAI-format tool schemas from a SkillContainer instance.

    Args:
        container: A SkillContainer instance with skills defined.

    Returns:
        List of OpenAI-compatible tool definitions.
    """
    tools: ToolSchemaList = []
    skills_dict = container.skills()

    for skill_name, skill_config in skills_dict.items():
        schema = skill_config.schema
        if schema.get("type") == "function":
            tools.append(schema)
            logger.debug(f"Extracted skill: {skill_name}")
        else:
            logger.warning(f"Invalid schema format for skill: {skill_name}")

    return tools


def extract_skills_from_container_class(
    container_class: type[SkillContainer],
) -> ToolSchemaList:
    """Extract OpenAI-format tool schemas from a SkillContainer class.

    Args:
        container_class: A SkillContainer class (not instance).

    Returns:
        List of OpenAI-compatible tool definitions.
    """
    try:
        container = container_class()
        return extract_skills_from_container(container)
    except Exception as e:
        logger.warning(f"Failed to instantiate {container_class.__name__}: {e}")
        return []


def extract_skills_from_blueprint(
    blueprint: ModuleBlueprintSet,
) -> ToolSchemaList:
    """Extract all skill schemas from modules in a blueprint.

    Args:
        blueprint: A ModuleBlueprintSet containing module blueprints.

    Returns:
        List of OpenAI-compatible tool definitions from all skill containers.
    """
    all_tools: ToolSchemaList = []
    seen_skills: set[str] = set()

    for module_blueprint in blueprint.blueprints:
        module_class = module_blueprint.module

        # Check if this module is a SkillContainer
        if not issubclass(module_class, SkillContainer):
            continue

        logger.info(f"Extracting skills from: {module_class.__name__}")

        try:
            # Instantiate the module with its blueprint args/kwargs
            instance = module_class(*module_blueprint.args, **module_blueprint.kwargs)
            tools = extract_skills_from_container(instance)

            # Deduplicate by skill name
            for tool in tools:
                skill_name = tool.get("function", {}).get("name", "")
                if skill_name and skill_name not in seen_skills:
                    all_tools.append(tool)
                    seen_skills.add(skill_name)
                elif skill_name in seen_skills:
                    logger.debug(f"Skipping duplicate skill: {skill_name}")

        except Exception as e:
            logger.warning(f"Failed to extract skills from {module_class.__name__}: {e}")
            continue

    logger.info(f"Extracted {len(all_tools)} unique skills from blueprint")
    return all_tools


def get_skill_summary(tools: ToolSchemaList) -> str:
    """Generate a human-readable summary of extracted skills.

    Args:
        tools: List of OpenAI-format tool definitions.

    Returns:
        Formatted string summary of all skills.
    """
    lines = [f"Extracted {len(tools)} skills:", "=" * 60]

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description").split("\n")[0][:60]
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"\n{name}")
        if desc:
            lines.append(f"  Description: {desc}...")
        lines.append("  Parameters:")
        for param_name, param_schema in params.items():
            req = " (required)" if param_name in required else " (optional)"
            param_type = param_schema.get("type", "unknown")
            lines.append(f"    - {param_name}: {param_type}{req}")

    return "\n".join(lines)
