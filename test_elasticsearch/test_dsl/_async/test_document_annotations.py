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

from datetime import date
from typing import Annotated, ClassVar, List, Optional

import pytest

from elasticsearch.dsl import (
    AsyncDocument,
    Date,
    InnerDoc,
    Keyword,
    M,
    Object,
    mapped_field,
)
from elasticsearch.dsl.exceptions import ValidationException


class Author(InnerDoc):
    name = Keyword()


def test_resolved_required_annotation() -> None:
    class Article(AsyncDocument):
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
        "Optional[str]",
        "M[Optional[str]]",
        '"str | None"',
        'Optional["str"]',
    ],
)
def test_resolved_optional_annotations(annotation: str) -> None:
    class Article(AsyncDocument):
        __annotations__ = {"note": annotation}
        note = mapped_field(Keyword(required=True, multi=True))

    assert Article().note is None
    Article().full_clean()


def test_resolved_mapped_list_annotation() -> None:
    class Article(AsyncDocument):
        tags: M[List[str]] = mapped_field(Keyword(required=True))

    assert Article().tags == []
    Article().full_clean()


def test_resolved_annotated_field() -> None:
    class Article(AsyncDocument):
        metadata: Annotated[Optional[str], Keyword(required=True)]

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"metadata": {"type": "keyword"}}
    }
    Article().full_clean()


def test_resolved_classvar_annotation() -> None:
    class Article(AsyncDocument):
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

    class Article(AsyncDocument):
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
    class Article(AsyncDocument):
        value: Annotated["Missing", Keyword(required=False, multi=True)]  # noqa: F821

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"value": {"type": "keyword"}}
    }
    assert Article().value == []
    Article().full_clean()


def test_self_reference_preserves_field() -> None:
    class Node(InnerDoc):
        parent: Optional[Node] = Keyword(required=False, multi=True)

    assert Node().parent == []
    Node().full_clean()


# Build the mapping before the referenced class exists in the module.
class ForwardArticle(AsyncDocument):
    author: Optional[LaterAuthor] = Keyword(required=False, multi=True)


class LaterAuthor(InnerDoc):
    name = Keyword()


def test_forward_reference_preserves_field() -> None:
    assert ForwardArticle().author == []
    ForwardArticle().full_clean()


def test_function_local_type_preserves_field() -> None:
    class Tag(InnerDoc):
        label: Optional[str] = Keyword()

    class Article(AsyncDocument):
        tag: Optional[Tag] = Object(Tag, required=False, multi=True)

    assert Article().tag == []
    Article().full_clean()

    with pytest.raises(TypeError, match="Cannot map field tag"):

        class ArticleWithoutField(AsyncDocument):
            tag: Optional[Tag]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("annotation", ["Optional[Missing]", 'M["Missing"]'])
def test_unresolved_annotation_without_field_raises(
    wrapped: bool, annotation: str
) -> None:
    with pytest.raises(TypeError, match="Cannot map field unknown") as exc:

        class Article(AsyncDocument):
            __annotations__ = {"unknown": annotation}
            unknown = mapped_field() if wrapped else None

    assert isinstance(exc.value.__cause__, NameError)
    assert "Missing" in str(exc.value.__cause__)


def test_unresolved_excluded_annotation() -> None:
    class Article(AsyncDocument):
        ignored: Missing = mapped_field(exclude=True)  # noqa: F821

    assert "ignored" not in Article._doc_type.mapping


def test_unresolved_annotation_preserves_mapped_field_options() -> None:
    class Article(AsyncDocument):
        note: Missing = mapped_field(  # noqa: F821
            Keyword(required=False, multi=True), default=["draft"], es_name="note_text"
        )

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"note_text": {"type": "keyword"}}
    }
    assert Article().to_dict() == {"note_text": ["draft"]}
    Article(note=None).full_clean()


def test_annotation_uses_class_namespace() -> None:
    class Article(AsyncDocument):
        Alias: ClassVar = Optional[str]
        note: Alias = Keyword(required=True)

    Article().full_clean()
    assert "Alias" not in Article._doc_type.mapping


def test_class_alias_takes_precedence_over_module_name() -> None:
    class Article(AsyncDocument):
        date: ClassVar = str
        value: date

    assert Article._doc_type.mapping.to_dict() == {
        "properties": {"value": {"type": "text"}}
    }


CircularAlias = "CircularAlias"
RecursiveAlias = List["RecursiveAlias"]


@pytest.mark.parametrize("annotation", ["CircularAlias", "RecursiveAlias"])
def test_circular_alias_preserves_field(annotation: str) -> None:
    class Article(AsyncDocument):
        __annotations__ = {"value": annotation}
        value = Keyword(required=False, multi=True)

    assert Article().value == []
    Article().full_clean()


@pytest.mark.parametrize("annotation", ["date", "M[date]", "Optional[date]"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_field_named_after_its_type(annotation: str, wrapped: bool) -> None:
    class Event(AsyncDocument):
        __annotations__ = {"date": annotation}
        date = (
            mapped_field(Date(format="yyyy-MM-dd"))
            if wrapped
            else Date(format="yyyy-MM-dd")
        )

    assert Event._doc_type.mapping.to_dict() == {
        "properties": {"date": {"type": "date", "format": "yyyy-MM-dd"}}
    }
    Event(date=date(2026, 1, 2)).full_clean()


def test_field_named_after_builtin_type() -> None:
    class Event(AsyncDocument):
        str: str = Keyword()

    assert Event._doc_type.mapping.to_dict() == {
        "properties": {"str": {"type": "keyword"}}
    }
    Event(str="value").full_clean()


@pytest.mark.parametrize("annotation", ["str | int | None", "Optional[str | int]"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_unsupported_union_preserves_field(annotation: str, wrapped: bool) -> None:
    explicit = Keyword(required=False, multi=True)

    class Article(AsyncDocument):
        __annotations__ = {"value": annotation}
        value = mapped_field(explicit) if wrapped else explicit

    assert Article._doc_type.mapping["value"] is explicit
    assert Article().value == []
    Article().full_clean()


def test_unsupported_union_without_field_raises() -> None:
    with pytest.raises(TypeError, match="Cannot map field value") as exc:

        class Article(AsyncDocument):
            value: str | int | None

    assert isinstance(exc.value.__cause__, TypeError)
    assert str(exc.value.__cause__) == "Unsupported union"


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
    def make_document() -> type[AsyncDocument]:
        class Article(AsyncDocument):
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
    class PlainArticle(AsyncDocument):
        __annotations__ = {
            "text": str,
            "created": date,
            "author": Author,
            "authors": List[Author],
        }

    class Article(AsyncDocument):
        text: str
        created: date
        author: Author
        authors: List[Author]

    expected = {
        "properties": {
            "text": {"type": "text"},
            "created": {"type": "date", "format": "yyyy-MM-dd"},
            "author": {"type": "object", "properties": {"name": {"type": "keyword"}}},
            "authors": {"type": "nested", "properties": {"name": {"type": "keyword"}}},
        }
    }
    assert PlainArticle._doc_type.mapping.to_dict() == expected
    assert Article._doc_type.mapping.to_dict() == expected
