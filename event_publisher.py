from collections.abc import Callable


class EventPublisher[T]:
    def __init__(self, subscribers: list[Callable] | None = None):

        if subscribers is None:
            self._subscribers: list[Callable] = []
        else:
            self._subscribers: list[Callable] = subscribers

    def publish(self, event: T) -> None:
        for subscriber in self._subscribers:
            subscriber(event)

    def subscribe(self, callback: Callable[T, None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[T, None]) -> None:
        self._subscribers.remove(callback)
