"""Endpoint mark and Edge property semantics."""

from cbcd.graph import Edge, EndpointMark


def test_endpoint_mark_values() -> None:
    assert EndpointMark.NO_EDGE == 0
    assert EndpointMark.TAIL == 1
    assert EndpointMark.ARROW == 2
    assert EndpointMark.CIRCLE == 3


def test_directed_edge() -> None:
    e = Edge(0, 1, EndpointMark.TAIL, EndpointMark.ARROW)
    assert e.is_directed
    assert not e.is_undirected
    assert not e.is_bidirected
    assert not e.is_circle_circle


def test_directed_edge_reversed() -> None:
    e = Edge(0, 1, EndpointMark.ARROW, EndpointMark.TAIL)
    assert e.is_directed


def test_undirected_edge() -> None:
    e = Edge(0, 1, EndpointMark.TAIL, EndpointMark.TAIL)
    assert e.is_undirected
    assert not e.is_directed


def test_bidirected_edge() -> None:
    e = Edge(0, 1, EndpointMark.ARROW, EndpointMark.ARROW)
    assert e.is_bidirected
    assert not e.is_directed
    assert not e.is_undirected


def test_circle_circle_edge() -> None:
    e = Edge(0, 1, EndpointMark.CIRCLE, EndpointMark.CIRCLE)
    assert e.is_circle_circle
    assert not e.is_directed
