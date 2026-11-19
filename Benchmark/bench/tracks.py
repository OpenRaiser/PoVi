from __future__ import annotations

from dataclasses import dataclass


ArtifactOptions = tuple[frozenset[str], ...]


@dataclass(frozen=True)
class TrackDefinition:
    track_id: str
    public_name: str
    settings: frozenset[str]
    required_artifacts_by_setting: dict[str, ArtifactOptions]
    main_domains_by_setting: dict[str, tuple[str, ...]]
    gate_ids_by_setting: dict[str, frozenset[str]]

    def required_artifacts_for(self, setting: str) -> ArtifactOptions:
        return self.required_artifacts_by_setting[setting]

    def main_domains_for(self, setting: str) -> tuple[str, ...]:
        return self.main_domains_by_setting[setting]

    def gate_ids_for(self, setting: str) -> frozenset[str]:
        return self.gate_ids_by_setting[setting]

    def requires_one_of(self, setting: str, artifacts: set[str]) -> bool:
        return any(
            option <= artifacts for option in self.required_artifacts_for(setting)
        )


@dataclass(frozen=True)
class Route:
    track: TrackDefinition
    setting: str

    @property
    def track_id(self) -> str:
        return self.track.track_id

    @property
    def public_name(self) -> str:
        return self.track.public_name

    @property
    def main_domains(self) -> tuple[str, ...]:
        return self.track.main_domains_for(self.setting)

    @property
    def gate_ids(self) -> frozenset[str]:
        return self.track.gate_ids_for(self.setting)

    @property
    def required_artifacts(self) -> ArtifactOptions:
        return self.track.required_artifacts_for(self.setting)

    def requires_one_of(self, artifacts: set[str]) -> bool:
        return self.track.requires_one_of(self.setting, artifacts)


TRACKS: dict[str, TrackDefinition] = {
    "text_planning": TrackDefinition(
        track_id="text_planning",
        public_name="Track 1 Text Planning",
        settings=frozenset({"standard"}),
        required_artifacts_by_setting={
            "standard": (
                frozenset({"planning_output"}),
                frozenset({"narration", "storyboard"}),
            ),
        },
        main_domains_by_setting={
            "standard": ("content",),
        },
        gate_ids_by_setting={
            "standard": frozenset({"G1.1", "G2.1"}),
        },
    ),
    "visual_generation": TrackDefinition(
        track_id="visual_generation",
        public_name="Track 2 Visual Generation",
        settings=frozenset({"oracle_conditioned", "system_conditioned"}),
        required_artifacts_by_setting={
            "oracle_conditioned": (frozenset({"final_video"}),),
            "system_conditioned": (frozenset({"final_video"}),),
        },
        main_domains_by_setting={
            "oracle_conditioned": ("visual",),
            "system_conditioned": ("visual",),
        },
        gate_ids_by_setting={
            "oracle_conditioned": frozenset({"G1.1", "G2.2"}),
            "system_conditioned": frozenset({"G1.1", "G2.2"}),
        },
    ),
    "audio_generation": TrackDefinition(
        track_id="audio_generation",
        public_name="Track 3 Audio Generation",
        settings=frozenset({"fixed_narration"}),
        required_artifacts_by_setting={
            "fixed_narration": (
                frozenset({"audio"}),
                frozenset({"final_video"}),
            )
        },
        # The formal A-domain metrics were removed. Track 3 remains as a
        # compatibility route that validates the submitted audio artifact.
        main_domains_by_setting={"fixed_narration": ()},
        gate_ids_by_setting={
            "fixed_narration": frozenset({"G1.1", "G2.3"})
        },
    ),
    "audio_visual_generation": TrackDefinition(
        track_id="audio_visual_generation",
        public_name="Track 4 Audio-Visual Generation",
        settings=frozenset({"structured_script"}),
        required_artifacts_by_setting={
            "structured_script": (frozenset({"final_video"}),),
        },
        main_domains_by_setting={
            "structured_script": ("visual", "av"),
        },
        gate_ids_by_setting={
            "structured_script": frozenset({"G1.1", "G2.2", "G2.3"}),
        },
    ),
    "full_task_teaching": TrackDefinition(
        track_id="full_task_teaching",
        public_name="Track 5 Full-Task Poetry Teaching Video",
        settings=frozenset({"full_task"}),
        required_artifacts_by_setting={
            "full_task": (frozenset({"final_video"}),)
        },
        main_domains_by_setting={
            "full_task": ("content", "visual", "av")
        },
        gate_ids_by_setting={
            "full_task": frozenset({"G1.1", "G2.2", "G2.3"})
        },
    ),
}

TRACK_IDS = frozenset(TRACKS)
SETTING_IDS = frozenset(
    setting
    for definition in TRACKS.values()
    for setting in definition.settings
)

def get_track(track_id: str) -> TrackDefinition:
    if track_id not in TRACKS:
        raise ValueError(f"Unknown track: {track_id}")
    return TRACKS[track_id]


def resolve_route(track_id: str, setting: str) -> Route:
    definition = get_track(track_id)
    if setting not in definition.settings:
        expected = ", ".join(sorted(definition.settings))
        raise ValueError(
            f"Setting {setting!r} is invalid for {track_id}; "
            f"expected one of {expected}"
        )
    return Route(track=definition, setting=setting)


def expected_domains(track_id: str, setting: str) -> tuple[str, ...]:
    return resolve_route(track_id, setting).main_domains
