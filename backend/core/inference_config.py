from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceProfile:
    beam_size: int
    temperature: float
    best_of: int
    condition_on_previous_text: bool
    vad_filter: bool
    initial_prompt: Optional[str]

# 1. Streaming Profile (Fast, English)
STREAMING_PROFILE = InferenceProfile(
    beam_size=2,
    temperature=0.0,
    best_of=1,
    condition_on_previous_text=False,
    vad_filter=True,
    initial_prompt=None
)

# 2. Accuracy Profile (Slower, context-aware)
ACCURACY_PROFILE = InferenceProfile(
    beam_size=5,
    temperature=0.0,
    best_of=3,
    condition_on_previous_text=True,
    vad_filter=True,
    initial_prompt=None
)

# 3. Urdu Profile (Native Script)
URDU_PROFILE = InferenceProfile(
    beam_size=5,
    temperature=0.0,
    best_of=3,
    condition_on_previous_text=True,
    vad_filter=True,
    initial_prompt="The following speech is in Urdu. Transcribe carefully."
)

# 4. Roman Urdu Profile (Desi/English Script)
ROMAN_URDU_PROFILE = InferenceProfile(
    beam_size=5,
    temperature=0.0,
    best_of=3,
    condition_on_previous_text=True,
    vad_filter=True,
    initial_prompt="The following speech is in Urdu. Write the output in Roman Urdu using English letters."
)

def get_profile_for_language(language: str) -> InferenceProfile:
    """Selects the best inference profile based on language code."""
    if language == "ur":
        return URDU_PROFILE
    elif language == "roman-ur": # Assuming "roman-ur" or empty string from frontend
        return ROMAN_URDU_PROFILE
    elif language == "en":
        return STREAMING_PROFILE
    else:
        # Default to accuracy profile for unknown languages or mixed use
        return ACCURACY_PROFILE
