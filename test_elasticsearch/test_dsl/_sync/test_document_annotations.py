#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from __future__ import annotations

import subprocess
import sys
from datetime import date
from typing import Annotated, ClassVar, List, Optional

import pytest

from elasticsearch.dsl import (
    Date,
    Document,
    InnerDoc,
    Keyword,
    M,
    Object,
    Text,
    mapped_field,
)
from elasticsearch.dsl.exceptions import ValidationException


class Author(InnerDoc):
    name = Keyword()


def test_resolved_required_annotation() -> None:
    class Article(Document):
        title: str = Keyword(required=False, multi=True)

    assert Article().title is None
    Article(title="test").full_clean()
    with pytest.raises(ValidationException) as exc:
        Article().full_clean()
    assert set(exc.value.args[0]) == {"title"}


@pytest.mark.parametrize(
    "annotation",
    [
        "str | None",
        "None | str",
        "Optional[str]",
        "M[Optional[str]]",
        '"str | None"',
        'Optional["str"]',
        "Optional[None | str]",
    ],
)
def test_resolved_optional_annotations(annotation: str) -> None:
    class Article(Document):
        __annotations__ = {"note": annotation}
        note = mapped_field(Keyword(required=True, multi=True))

    assert Article().note is None
    Article().full_clean()


def test_none_first_optional_infers_field() -> None:
    class Article(Document):
        note: None | str

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"note": {"type": "text"}}
    }
    assert Article().note is None
    Article().full_clean()
    Article(note="draft").full_clean()


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("postponed", [False, True])
def test_optional_union_does_not_depend_on_typing_cache(
    warm_cache: bool, postponed: bool
) -> None:
    # Isolate typing's cache so other tests cannot determine the union order.
    source = "from __future__ import annotations\n" if postponed else ""
    source += f"""
import sys
from typing import Optional, Union
from elasticsearch.dsl import {Document.__name__}, Keyword

if sys.argv[1] == "warm":
    Optional[Union[None, str]]

class Article({Document.__name__}):
    note: Optional[Optional[str]] = Keyword(required=True)

Article().full_clean()
"""
    result = subprocess.run(
        [sys.executable, "-c", source, "warm" if warm_cache else "cold"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_resolved_mapped_list_annotation() -> None:
    class Article(Document):
        tags: M[List[str]] = mapped_field(Keyword(required=True))

    assert Article().tags == []
    Article().full_clean()


def test_resolved_annotated_field() -> None:
    class Article(Document):
        metadata: Annotated[Optional[str], Keyword(required=True)]

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"metadata": {"type": "keyword"}}
    }
    Article().full_clean()


@pytest.mark.parametrize(
    "wrapper, metadata, field_type, required, multi",
    [
        ("M", "Keyword(required=False, multi=True)", "keyword", True, False),
        ("List", "Keyword(required=False, multi=True)", "keyword", False, True),
        ("Optional", '"marker"', "text", False, False),
        pytest.param(
            "Optional",
            "Keyword(required=False, multi=True)",
            "keyword",
            False,
            False,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 11),
                reason="Python 3.10 unions reject unhashable Annotated metadata",
            ),
        ),
    ],
)
def test_nested_annotated_field(
    wrapper: str, metadata: str, field_type: str, required: bool, multi: bool
) -> None:
    class Article(Document):
        __annotations__ = {"value": f"{wrapper}[Annotated[str, {metadata}]]"}

    field = Article._doc_type.mapping["value"]
    assert field.to_dict() == {"type": field_type}
    assert (field._required, field._multi) == (required, multi)
    assert Article().value == ([] if multi else None)
    if required:
        with pytest.raises(ValidationException):
            Article().full_clean()
    else:
        Article().full_clean()
    Article(value="test").full_clean()


@pytest.mark.parametrize(
    "annotation",
    [
        "M[Annotated[str, Keyword()]]",
        "Optional[Annotated[str, Keyword()]]",
        "List[Annotated[str, Keyword()]]",
        "M[Annotated[str, mapped_field(exclude=True)]]",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_explicit_field_takes_precedence_over_nested_metadata(
    annotation: str, wrapped: bool
) -> None:
    explicit = Text()
    default = ["draft"] if annotation.startswith("List[") else "draft"

    class Article(Document):
        __annotations__ = {"title": annotation}
        title = (
            mapped_field(explicit, default=default, es_name="title_text")
            if wrapped
            else explicit
        )

    assert Article._doc_type.mapping["title"] is explicit
    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"title_text" if wrapped else "title": {"type": "text"}}
    }
    if wrapped:
        assert Article().to_dict() == {"title_text": default}


@pytest.mark.parametrize(
    "metadata",
    ["Keyword()", "mapped_field(Keyword(), default='metadata', es_name='ignored')"],
)
def test_mapped_field_options_keep_the_annotation_field(metadata: str) -> None:
    class Article(Document):
        __annotations__ = {"title": f"M[Annotated[str, {metadata}]]"}
        title = mapped_field(default="draft", es_name="title_text")

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"title_text": {"type": "keyword"}}
    }
    assert Article().to_dict() == {"title_text": "draft"}


def test_nullable_list_infers_multiple_values() -> None:
    class Article(Document):
        tags: list[str] | None

    field = Article._doc_type.mapping["tags"]
    assert (field._required, field._multi) == (False, True)
    article = Article()
    assert article.tags == []
    article.tags.append("tag")
    assert article.to_dict() == {"tags": ["tag"]}
    article.full_clean()


def test_resolved_classvar_annotation() -> None:
    class Article(Document):
        ignored: ClassVar[str] = "not a field"

    assert Article.ignored == "not a field"
    assert Article._doc_type.mapping.to_dict() == {}


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "annotation",
    [
        "Optional[Missing]",
        'Optional["Missing"]',
        'List["Missing"]',
        'list["Missing"]',
        'M["Missing"]',
    ],
)
def test_unresolved_annotation_preserves_field(
    required: bool, multi: bool, wrapped: bool, annotation: str
) -> None:
    explicit = Keyword(required=required, multi=multi)

    class Article(Document):
        __annotations__ = {"unknown": annotation, "known": "Optional[str]"}
        unknown = mapped_field(explicit) if wrapped else explicit
        known = Keyword(required=True, multi=True)

    assert Article._doc_type.mapping["unknown"] is explicit
    assert Article().unknown == ([] if multi else None)
    assert Article().known is None
    if required:
        with pytest.raises(ValidationException) as exc:
            Article().full_clean()
        assert set(exc.value.args[0]) == {"unknown"}
    else:
        Article().full_clean()


def test_unresolved_annotated_field_preserves_settings() -> None:
    class Article(Document):
        value: Annotated["Missing", Keyword(required=False, multi=True)]  # noqa: F821

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"value": {"type": "keyword"}}
    }
    assert Article().value == []
    Article().full_clean()


def test_unquoted_unresolved_annotated_type_preserves_explicit_field() -> None:
    class Article(Document):
        value: Annotated[Missing, "metadata"] = Keyword(  # noqa: F821
            required=False, multi=True
        )

    assert Article().value == []
    Article().full_clean()


def test_self_reference_preserves_field() -> None:
    class Node(InnerDoc):
        parent: Optional[Node] = Keyword(required=False, multi=True)

    assert Node().parent == []
    Node().full_clean()


# Build the mapping before the referenced class exists in the module.
class ForwardArticle(Document):
    author: Optional[LaterAuthor] = Keyword(required=False, multi=True)


class LaterAuthor(InnerDoc):
    name = Keyword()


def test_forward_reference_preserves_field() -> None:
    assert ForwardArticle().author == []
    ForwardArticle().full_clean()


def test_function_local_type_preserves_field() -> None:
    class Tag(InnerDoc):
        label: Optional[str] = Keyword()

    class Article(Document):
        tag: Optional[Tag] = Object(Tag, required=False, multi=True)

    assert Article().tag == []
    Article().full_clean()

    with pytest.raises(TypeError, match="Cannot map field tag"):

        class ArticleWithoutField(Document):
            tag: Optional[Tag]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("annotation", ["Optional[Missing]", 'M["Missing"]'])
def test_unresolved_annotation_without_field_raises(
    wrapped: bool, annotation: str
) -> None:
    with pytest.raises(TypeError, match="Cannot map field unknown") as exc:

        class Article(Document):
            __annotations__ = {"unknown": annotation}
            unknown = mapped_field() if wrapped else None

    assert isinstance(exc.value.__cause__, NameError)
    assert "Missing" in str(exc.value.__cause__)


def test_unresolved_excluded_annotation() -> None:
    class Article(Document):
        ignored: Missing = mapped_field(exclude=True)  # noqa: F821

    assert "ignored" not in Article._doc_type.mapping


@pytest.mark.parametrize("annotation", ["M[int | str]", "List", "1 / 0"])
def test_excluded_annotation_is_not_evaluated(
    annotation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def resolve(*args: object) -> None:
        calls.append(args)
        raise AssertionError("Excluded annotations must not be resolved")

    monkeypatch.setattr("elasticsearch.dsl.document_base._resolve_annotation", resolve)

    class Article(Document):
        __annotations__ = {"ignored": annotation}
        ignored = mapped_field(exclude=True)

    assert "ignored" not in Article._doc_type.mapping
    assert calls == []


@pytest.mark.parametrize(
    "annotation",
    [
        "Annotated[int | str, mapped_field(exclude=True)]",
        "M[Annotated[int | str, mapped_field(exclude=True)]]",
    ],
)
def test_excluded_annotation_metadata_skips_inference(annotation: str) -> None:
    class Article(Document):
        __annotations__ = {"ignored": annotation}

    assert "ignored" not in Article._doc_type.mapping


def test_unresolved_annotation_preserves_mapped_field_options() -> None:
    class Article(Document):
        note: Missing = mapped_field(  # noqa: F821
            Keyword(required=False, multi=True), default=["draft"], es_name="note_text"
        )

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"note_text": {"type": "keyword"}}
    }
    assert Article().to_dict() == {"note_text": ["draft"]}
    Article(note=None).full_clean()


def test_annotation_uses_class_namespace() -> None:
    class Article(Document):
        Alias: ClassVar = Optional[str]
        note: Alias = Keyword(required=True)

    Article().full_clean()
    assert "Alias" not in Article._doc_type.mapping


def test_class_alias_takes_precedence_over_module_name() -> None:
    class Article(Document):
        date: ClassVar = str
        value: date

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"value": {"type": "text"}}
    }


CircularAlias = "CircularAlias"
RecursiveAlias = List["RecursiveAlias"]


@pytest.mark.parametrize("annotation", ["CircularAlias", "RecursiveAlias"])
def test_circular_alias_preserves_field(annotation: str) -> None:
    class Article(Document):
        __annotations__ = {"value": annotation}
        value = Keyword(required=False, multi=True)

    assert Article().value == []
    Article().full_clean()


@pytest.mark.parametrize("annotation", ["date", "M[date]", "Optional[date]"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_field_named_after_its_type(annotation: str, wrapped: bool) -> None:
    class Event(Document):
        __annotations__ = {"date": annotation}
        date = (
            mapped_field(Date(format="yyyy-MM-dd"))
            if wrapped
            else Date(format="yyyy-MM-dd")
        )

    assert Event._doc_type.mapping.to_dict() == {
        "properties": {"date": {"type": "date", "format": "yyyy-MM-dd"}}
    }
    assert Event._doc_type.mapping["date"]._required == (annotation != "Optional[date]")
    assert Event._doc_type.mapping["date"]._multi is False
    Event(date=date(2026, 1, 2)).full_clean()


def test_field_named_after_builtin_type() -> None:
    class Event(Document):
        str: str = Keyword()

    assert Event._doc_type.mapping.to_dict() == {
        "properties": {"str": {"type": "keyword"}}
    }
    Event(str="value").full_clean()


@pytest.mark.parametrize(
    "member_kind", ["method", "property", "classmethod", "staticmethod"]
)
def test_class_members_do_not_shadow_annotation_types(member_kind: str) -> None:
    class Event(Document):
        day: date

        def date(self) -> None:
            pass

        if member_kind == "property":
            date = property(date)
        elif member_kind == "classmethod":
            date = classmethod(date)
        elif member_kind == "staticmethod":
            date = staticmethod(date)

    assert Event._doc_type.mapping.to_dict() == {
        "properties": {"day": {"type": "date", "format": "yyyy-MM-dd"}}
    }
    Event(day=date(2026, 1, 2)).full_clean()


@pytest.mark.parametrize(
    "annotation", ["str | int", "str | int | None", "Optional[str | int]", "List"]
)
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("required, multi", [(False, True), (True, False)])
def test_unsupported_string_annotation_preserves_field(
    annotation: str, wrapped: bool, required: bool, multi: bool
) -> None:
    explicit = Keyword(required=required, multi=multi)

    class Article(Document):
        __annotations__ = {"value": annotation}
        value = mapped_field(explicit) if wrapped else explicit

    assert Article._doc_type.mapping["value"] is explicit
    assert (explicit._required, explicit._multi) == (required, multi)
    assert Article().value == ([] if multi else None)
    if required:
        with pytest.raises(ValidationException):
            Article().full_clean()
    else:
        Article().full_clean()


@pytest.mark.parametrize("annotation", ["str | int | None", "List"])
def test_unsupported_string_annotation_without_field_raises(annotation: str) -> None:
    with pytest.raises(TypeError, match="Cannot map field value") as exc:

        class Article(Document):
            __annotations__ = {"value": annotation}

    assert isinstance(exc.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "annotation, error_type",
    [
        ('int("invalid")', ValueError),
        ('{}["missing"]', KeyError),
        ('__import__("_missing_annotation_module_3554")', ModuleNotFoundError),
        ("List[int, str]", TypeError),
        ("date.missing", AttributeError),
        ("str |", SyntaxError),
    ],
)
@pytest.mark.parametrize("with_field", [False, True])
def test_annotation_evaluation_errors(
    annotation: str, error_type: type[Exception], with_field: bool
) -> None:
    def make_document() -> type[Document]:
        class Article(Document):
            __annotations__ = {"value": annotation}
            value = Keyword(required=False, multi=True) if with_field else None

        return Article

    if with_field:
        document = make_document()
        assert document().value == []
        document().full_clean()
    else:
        with pytest.raises(TypeError, match="Cannot map field value") as exc:
            make_document()
        assert isinstance(exc.value.__cause__, error_type)


def test_future_annotations_have_same_mapping_as_plain_annotations() -> None:
    class PlainArticle(Document):
        __annotations__ = {
            "text": str,
            "created": date,
            "author": Author,
            "authors": List[Author],
            "note": Optional[str],
        }

    class Article(Document):
        text: str
        created: date
        author: Author
        authors: List[Author]
        note: Optional[str]

    expected = {
        "properties": {
            "text": {"type": "text"},
            "created": {"type": "date", "format": "yyyy-MM-dd"},
            "author": {"type": "object", "properties": {"name": {"type": "keyword"}}},
            "authors": {"type": "nested", "properties": {"name": {"type": "keyword"}}},
            "note": {"type": "text"},
        }
    }
    assert PlainArticle._doc_type.mapping.to_dict() == expected
    assert Article._doc_type.mapping.to_dict() == expected
    for name, settings in {
        "text": (True, False),
        "created": (True, False),
        "author": (True, False),
        "authors": (False, True),
        "note": (False, False),
    }.items():
        for document in (PlainArticle, Article):
            field = document._doc_type.mapping[name]
            assert (field._required, field._multi) == settings
