# SPDX-FileCopyrightText: 2026-present Monolith Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Wire-level tests: the real shopsavvy SDK client (request building + pydantic
parsing) and real Haystack components/pipelines, with only the network swapped
for an httpx.MockTransport serving the Data API's wire shape. Unlike the
MagicMock-based tests, these fail if a component reads a field the SDK models
do not have.
"""

import json

import httpx
import pytest
from haystack import Pipeline
from haystack.utils import Secret
from shopsavvy import RateLimitError

from haystack_integrations.components.converters.shopsavvy import ShopSavvyPriceComparison, ShopSavvyProductSearch

PRODUCT = {
    "title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
    "shopsavvy": "sp_456",
    "brand": "Sony",
    "category": "Headphones",
    "barcode": "027242923782",
    "amazon": "B09XS7JWHH",
    "model": "WH1000XM5/B",
    "description": "Industry-leading noise canceling headphones.",
}

OFFERS = [
    {
        "id": "of_1",
        "retailer": "Amazon",
        "price": 298.0,
        "currency": "USD",
        "availability": "in",
        "condition": "new",
        "URL": "https://www.amazon.com/dp/B09XS7JWHH",
        "seller": "Amazon.com",
        "timestamp": "2026-09-01T12:00:00Z",
    },
    {
        "id": "of_2",
        "retailer": "Best Buy",
        "price": 329.99,
        "currency": "USD",
        "availability": "in",
        "condition": "new",
        "URL": "https://www.bestbuy.com/site/6505727.p",
        "timestamp": "2026-09-01T11:00:00Z",
    },
]


def _api(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/products/search":
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": [PRODUCT],
                "pagination": {"total": 1, "limit": 3, "offset": 0, "returned": 1},
            },
        )
    if request.url.path == "/v1/products/offers":
        return httpx.Response(200, json={"success": True, "data": [{**PRODUCT, "offers": OFFERS}]})
    return httpx.Response(404, json={"error": "not found"})


def _wired(component, handler=_api, seen=None):
    component.warm_up()
    real = component._client._client

    def record(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)

    component._client._client = httpx.Client(
        base_url=real.base_url, headers=real.headers, transport=httpx.MockTransport(record)
    )
    return component


KEY = Secret.from_token("ss_test_wire_tests")


def test_product_search_wire():
    seen = []
    search = _wired(ShopSavvyProductSearch(api_key=KEY, top_k=3), seen=seen)

    docs = search.run(query="sony headphones")["documents"]

    assert seen[0].url.params["q"] == "sony headphones"
    assert seen[0].url.params["limit"] == "3"
    assert seen[0].headers["authorization"] == "Bearer ss_test_wire_tests"
    assert len(docs) == 1
    assert json.loads(docs[0].content)["description"] == PRODUCT["description"]
    assert docs[0].meta == {
        "title": PRODUCT["title"],
        "brand": "Sony",
        "category": "Headphones",
        "shopsavvy_id": "sp_456",
        "barcode": "027242923782",
        "asin": "B09XS7JWHH",
        "source": "shopsavvy",
    }


def test_price_comparison_wire():
    seen = []
    compare = _wired(ShopSavvyPriceComparison(api_key=KEY, retailer="amazon.com"), seen=seen)

    docs = compare.run(identifier="B09XS7JWHH")["documents"]

    assert seen[0].url.path == "/v1/products/offers"
    assert seen[0].url.params["ids"] == "B09XS7JWHH"
    assert seen[0].url.params["retailer"] == "amazon.com"
    assert [d.meta["retailer"] for d in docs] == ["Amazon", "Best Buy"]
    assert docs[0].meta["price"] == 298.0
    assert docs[0].meta["url"] == "https://www.amazon.com/dp/B09XS7JWHH"
    assert json.loads(docs[0].content)["seller"] == "Amazon.com"
    assert json.loads(docs[1].content)["seller"] is None


def test_rate_limit_propagates():
    search = _wired(
        ShopSavvyProductSearch(api_key=KEY),
        handler=lambda _request: httpx.Response(429, json={"error": "slow down"}),
    )
    with pytest.raises(RateLimitError):
        search.run(query="anything")


def test_pipeline_serialization_round_trip(monkeypatch):
    monkeypatch.setenv("SHOPSAVVY_API_KEY", "ss_test_env_key")
    pipeline = Pipeline()
    pipeline.add_component("search", ShopSavvyProductSearch(top_k=3))
    pipeline.add_component("compare", ShopSavvyPriceComparison(retailer="amazon.com"))

    restored = Pipeline.loads(pipeline.dumps())

    search = restored.get_component("search")
    compare = restored.get_component("compare")
    assert search.top_k == 3
    assert compare.retailer == "amazon.com"
    assert search.api_key.resolve_value() == "ss_test_env_key"


def test_readme_pipeline_example_runs():
    pipeline = Pipeline()
    pipeline.add_component("search", _wired(ShopSavvyProductSearch(api_key=KEY, top_k=5)))
    pipeline.add_component("compare", _wired(ShopSavvyPriceComparison(api_key=KEY)))

    result = pipeline.run(
        {
            "search": {"query": "sony wh-1000xm5"},
            "compare": {"identifier": "B09XS7JWHH"},
        }
    )

    assert result["search"]["documents"][0].meta["asin"] == "B09XS7JWHH"
    assert [d.meta["price"] for d in result["compare"]["documents"]] == [298.0, 329.99]
