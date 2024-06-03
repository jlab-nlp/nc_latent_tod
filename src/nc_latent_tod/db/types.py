from typing import TypedDict


class DBEntity(TypedDict):
    """
    Base DB entity returned by some DB implementation
    """
    pass


class MultiWOZRestaurant(DBEntity):
    """
    A restaurant in MultiWOZ DB (as in Tomiinek/MultiWOZ_Evaluation)
    """
    address: str
    area: str
    food: str
    id: str
    name: str
    phone: str
    postcode: str
    pricerange: str
    type: str


class MultiWOZAttraction(DBEntity):
    """
    An attraction in MultiWOZ DB (as in Tomiinek/MultiWOZ_Evaluation)
    """
    address: str
    area: str
    entrancefee: str
    id: str
    name: str
    phone: str
    postcode: str
    pricerange: str
    type: str


class MultiWOZHotel(DBEntity):
    """
    A hotel in MultiWOZ DB (as in Tomiinek/MultiWOZ_Evaluation)
    """
    address: str
    area: str
    id: str
    internet: str
    name: str
    parking: str
    phone: str
    postcode: str
    pricerange: str
    stars: str
    type: str


class MultiWOZTrain(DBEntity):
    """
    A train in MultiWOZ DB (as in Tomiinek/MultiWOZ_Evaluation)
    """
    arrive: str
    day: str
    departure: str
    destination: str
    duration: str
    leave: str
    price: str
    trainid: str
