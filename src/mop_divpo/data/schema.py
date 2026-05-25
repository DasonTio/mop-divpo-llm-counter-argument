from dataclasses import dataclass, field


@dataclass
class SFTRecord:
    messages: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"messages": self.messages, "metadata": self.metadata}
