"""
William Phomphakdee
CSE 163

This file contains code for an event-based pattern, the publisher
subscriber model.
"""

from collections.abc import Callable


class EventPublisher[T]:
    """
    A class defining a central point to publish events to and have
    subscribers execute on that event.
    """
    def __init__(self, subscribers: list[Callable] | None = None):
        """
        Define an event publisher based on a pre-existing list of subscribers
        or an empty list of subscribers.
        :param subscribers: A list of callables that will be executed on
        """
        if subscribers is None:
            self._subscribers: list[Callable] = []
        else:
            self._subscribers: list[Callable] = subscribers

    def publish(self, event: T) -> None:
        """
        Publish a new event to the subscribers.
        :param event: an event which will be published
        :return: None
        """
        for subscriber in self._subscribers:
            subscriber(event)

    def subscribe(self, callback: Callable[T, None]) -> None:
        """
        Add a subscriber callback to this publisher's list of subscribers.
        :param callback: subscriber to add
        :return: None
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[T, None]) -> None:
        """
        Remove a subscriber callback from this publisher's list of subscribers.
        :param callback: subscriber to remove
        :return: None
        """
        self._subscribers.remove(callback)

    def get_subscribers(self) -> list[Callable]:
        """
        Return a list of all subscribers this publisher has registered.
        :return: list of subscribers
        """
        return self._subscribers
