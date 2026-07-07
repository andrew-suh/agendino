from pydantic import BaseModel


class EnrollSpeakerRequestDTO(BaseModel):
    recording_name: str
    speaker_label: str  # display label in the transcript, e.g. "Speaker 1"
    person_name: str
