"""01: 保证一个 Manifest 只包含同一 run 且 sample_id 唯一的记录。"""

from dataclasses import asdict, dataclass, field

@dataclass
class SampleRecord:
    run_id: str
    sample_id: str
    prompt: str


@dataclass
class SampleManifest:
    run_id: str
    records: list[SampleRecord] = field(default_factory=list)

    def add(self, records: SampleRecord) -> None:
        # FILL_ME 1: run_id 不一致时抛出 ValueError。
        # FILL_ME 2: sample_id 已存在时抛出 ValueError。
        if self.run_id is None or self.run_id.split() == "":
            raise ValueError("Invalid run_ids!")

        for record in records:
            if self.run_id != record.run_id:
                raise ValueError("run_id doesn't match")
            if any(record.sample_id == item.sample_id for item in records):
                raise ValueError("duplicate sample_ids")

            self.records.append(record)


    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleManifest":
        # FILL_ME 3: 先把 records 中的 dict 变成 SampleRecord，再构造 cls。
        records = [
            SampleRecord(**record_data)
            for record_data in data.get("records", [])
        ]
        raise NotImplementedError("FILL_ME")


def check() -> None:
    manifest = SampleManifest("run-a")
    manifest.add(SampleRecord("run-a", "sample-0", "red square"))
    restored = SampleManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    try:
        manifest.add(SampleRecord("run-b", "sample-1", "blue square"))
    except ValueError:
        pass
    else:
        raise AssertionError("run_id mismatch was not rejected")
    try:
        manifest.add(SampleRecord("run-a", "sample-0", "duplicate"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate sample_id was not rejected")
    print("01 PASS")


if __name__ == "__main__":
    check()
