import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from meshagent.agents.images_dataset import ImageDatasetRecord
from meshagent.api.messaging import FileContent

from meshagent.openai.tools.image_generation import (
    ImageGenerationToolkit,
    _ImageGenerationBackend,
)


def _image_bytes(image_format: str = "WEBP") -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (2, 3), (20, 40, 60, 128)).save(output, format=image_format)
    return output.getvalue()


class _FakeImagesApi:
    def __init__(self, generated: bytes) -> None:
        self.generated = generated
        self.generate_calls: list[dict] = []
        self.edit_calls: list[dict] = []

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(self.generated).decode("ascii")
                )
            ]
        )

    async def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(self.generated).decode("ascii")
                )
            ]
        )


class _FakeClient:
    def __init__(self, generated: bytes) -> None:
        self.images = _FakeImagesApi(generated)


class _FakeImagesDataset:
    def __init__(self) -> None:
        self.records: dict[str, ImageDatasetRecord] = {}
        self.saved: list[dict] = []
        self.deleted: list[str] = []

    async def read_record(self, *, image_id: str):
        return self.records.get(image_id)

    async def save(self, **kwargs):
        self.saved.append(kwargs)
        return SimpleNamespace(
            id=f"saved-{len(self.saved)}", mime_type=kwargs["mime_type"]
        )

    async def delete(self, *, image_id: str):
        self.deleted.append(image_id)
        return image_id in self.records


class _FakeStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def read_file(self, *, path: str):
        return FileContent(
            data=self.files[path],
            name=path.rsplit("/", maxsplit=1)[-1],
            mime_type="application/octet-stream",
        )

    async def write_bytes(self, *, path: str, data: bytes, overwrite: bool) -> None:
        assert overwrite
        self.files[path] = data


@pytest.mark.asyncio
async def test_image_generation_saves_generated_and_referenced_edits() -> None:
    images = _FakeImagesDataset()
    images.records["reference"] = ImageDatasetRecord(
        data=_image_bytes("PNG"),
        mime_type="image/png",
    )
    client = _FakeClient(_image_bytes("PNG"))
    backend = _ImageGenerationBackend(
        images_dataset=images,
        storage_toolkit=_FakeStorage(),
        client=client,
        model="gpt-image-test",
    )

    generated = await backend.generate(
        prompt="draw a fox",
        referenced_image_ids=None,
        created_by="agent-1",
    )
    edited = await backend.generate(
        prompt="make it blue",
        referenced_image_ids=["reference"],
        created_by="agent-1",
    )

    assert generated == {"saved_image_id": "saved-1", "mime_type": "image/png"}
    assert edited == {"saved_image_id": "saved-2", "mime_type": "image/png"}
    assert client.images.generate_calls[0]["model"] == "gpt-image-test"
    assert "response_format" not in client.images.generate_calls[0]
    assert client.images.edit_calls[0]["image"][0][0] == "reference.png"
    assert "response_format" not in client.images.edit_calls[0]


@pytest.mark.asyncio
async def test_image_import_and_export_convert_unsupported_extensions_to_png() -> None:
    images = _FakeImagesDataset()
    images.records["source"] = ImageDatasetRecord(
        data=_image_bytes(),
        mime_type="image/webp",
    )
    storage = _FakeStorage()
    storage.files["/incoming/photo.webp"] = _image_bytes()
    backend = _ImageGenerationBackend(
        images_dataset=images,
        storage_toolkit=storage,
        client=_FakeClient(_image_bytes("PNG")),
        model="gpt-image-test",
    )

    exported = await backend.export(
        image_id="source",
        destination_path="/out/photo.webp",
    )
    exported_jpeg = await backend.export(
        image_id="source",
        destination_path="/out/photo.jpg",
    )
    imported = await backend.import_image(
        source_path="/incoming/photo.webp",
        created_by="agent-1",
    )

    assert exported["destination_path"] == "/out/photo.png"
    assert exported["mime_type"] == "image/png"
    assert exported_jpeg["destination_path"] == "/out/photo.jpg"
    assert exported_jpeg["mime_type"] == "image/jpeg"
    assert imported["saved_image_id"] == "saved-1"
    assert images.saved[0]["mime_type"] == "image/png"
    with Image.open(io.BytesIO(storage.files["/out/photo.png"])) as exported_image:
        assert exported_image.format == "PNG"
    with Image.open(io.BytesIO(storage.files["/out/photo.jpg"])) as exported_image:
        assert exported_image.format == "JPEG"


@pytest.mark.asyncio
async def test_read_and_delete_image_use_dataset_content_and_id() -> None:
    images = _FakeImagesDataset()
    images.records["source"] = ImageDatasetRecord(
        data=_image_bytes("PNG"),
        mime_type="image/png",
    )
    backend = _ImageGenerationBackend(
        images_dataset=images,
        storage_toolkit=_FakeStorage(),
        client=_FakeClient(_image_bytes("PNG")),
        model="gpt-image-test",
    )

    content = await backend.read(image_id="source")
    deleted = await backend.delete(image_id="source")

    assert isinstance(content, FileContent)
    assert content.name == "source.png"
    assert content.mime_type == "image/png"
    assert deleted == {"deleted_image_id": "source"}
    assert images.deleted == ["source"]


def test_image_generation_toolkit_uses_dataset_backed_tool_names() -> None:
    toolkit = ImageGenerationToolkit(
        images_dataset=_FakeImagesDataset(),
        storage_toolkit=_FakeStorage(),
        client=_FakeClient(_image_bytes("PNG")),
    )

    assert toolkit.name == "image-generation"
    assert [tool.name for tool in toolkit.tools] == [
        "imagegen",
        "read_image",
        "delete_image",
        "export_image",
        "import_image",
    ]
    for tool in toolkit.tools:
        assert set(tool.input_schema["required"]) == set(
            tool.input_schema["properties"]
        )
    assert toolkit.tools[0].input_schema["properties"]["referenced_image_ids"] == {
        "type": ["array", "null"],
        "items": {"type": "string"},
        "description": (
            "Image IDs from the images dataset to edit. "
            "Use an empty array to generate a new image."
        ),
    }
    assert toolkit.tools[0].input_schema["required"] == [
        "prompt",
        "referenced_image_ids",
    ]
    assert toolkit.tools[1].input_schema["required"] == ["id"]
