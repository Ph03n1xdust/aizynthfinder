""" Module containing classes that interfaces different stocks and query classes
"""
from __future__ import annotations
import os
import copy
import importlib
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from aizynthfinder.chem import Molecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.utils.mongo import get_mongo_client

if TYPE_CHECKING:
    from pymongo.database import Database as MongoDatabase
    from pymongo.collection import Collection as MongoCollection

    from aizynthfinder.utils.type_utils import StrDict, Set, Union, Any, Optional, List


class StockException(Exception):
    """An exception raised by the Stock classes"""


class StockQueryMixin:
    """
    Mixin class for all query classes, providing a default interface
    to some methods that might not be possible to implement for each
    query class.
    """

    def __len__(self) -> int:
        return 0

    def __contains__(self, mol: Molecule) -> bool:
        return False

    def amount(self, mol: Molecule) -> float:
        """
        Returns the maximum amount of the molecule in stock

        :param mol: the query molecule
        :raises StockException: if the amount cannot be computed
        :return: the amount
        """
        raise StockException("Cannot compute amount")

    def availability_string(self, mol: Molecule) -> str:
        """
        Returns the sources of the molecule

        :param mol: the query molecule
        :raises StockException: if the string cannot be computed
        :return: a comma-separated list of sources
        """
        raise StockException("Cannot provide availability")

    def cached_search(self, mol: Molecule) -> bool:
        """
        Finds the entries of the molecule in the stock and cache them
        if necessary.

        :param mol: the query molecule
        :return: if the molecule is in stock
        """
        return mol in self

    def clear_cache(self) -> None:
        """Clear the internal search cache if available"""
        pass

    def price(self, mol: Molecule) -> float:
        """
        Returns the minimum price of the molecule in stock

        :param mol: the query molecule
        :raises StockException: if the price cannot be computed
        :rtype: float
        """
        raise StockException("Cannot compute price")


class InMemoryInchiKeyQuery(StockQueryMixin):
    """
    A stock query class that is based on an in-memory list
    of pre-computed inchi-keys.

    This list can be instantiated from
        * A Pandas dataframe in HDF5 or CSV format
        * A text file with an inchi key on each row

    The pandas dataframe must have a column "inchi_key" with InChIkeys.
    The HDF5 file must have a dataset called "table".

    :parameter filename: the path to the file with inchi-keys
    """

    def __init__(self, filename: str) -> None:
        ext = os.path.splitext(filename)[1]
        if ext in [".h5", ".hdf5"]:
            stock = pd.read_hdf(filename, key="table")  # type: ignore
            inchis = stock.inchi_key.values  # type: ignore
        elif ext == ".csv":
            stock = pd.read_csv(filename)
            inchis = stock.inchi_key.values
        else:
            with open(filename, "r") as fileobj:
                inchis = fileobj.read().splitlines()
        self._stock_inchikeys = set(inchis)

    def __contains__(self, mol: Molecule) -> bool:
        return mol.inchi_key in self._stock_inchikeys

    def __len__(self) -> int:
        return len(self._stock_inchikeys)

    @property
    def stock_inchikeys(self) -> Set[str]:
        return self._stock_inchikeys


class MongoDbInchiKeyQuery(StockQueryMixin):
    """
    A stock query class that is looking up inchi keys in a Mongo database.

    The documents in the database collection should have at least 2 fields:
        * inchi_key: the inchi key of the molecule
        * source: the original source of the molecule

    :ivar client: the Mongo client
    :ivar database: the database instance
    :ivar molecules: the collection of documents

    :parameter host: the database host, defaults to None
    :parameter database: the database name, defaults to "stock_db"
    :parameter collection: the database collection, defaults to "molecules"
    """

    def __init__(
        self,
        host: str = None,
        database: str = "stock_db",
        collection: str = "molecules",
    ) -> None:
        self.client = get_mongo_client(
            host or os.environ.get("MONGODB_HOST") or "localhost"
        )
        self.database: MongoDatabase = self.client[database]
        self.molecules: MongoCollection = self.database[collection]
        self._len: Optional[int] = None

    def __contains__(self, mol: Molecule) -> bool:
        return self.molecules.count_documents({"inchi_key": mol.inchi_key}) > 0

    def __len__(self) -> int:
        if self._len is None:
            self._len = self.molecules.count_documents({})
        return self._len

    def __str__(self) -> str:
        return "'MongoDB stock'"

    def availability_string(self, mol: Molecule) -> str:
        sources = [
            item["source"] for item in self.molecules.find({"inchi_key": mol.inchi_key})
        ]
        return ",".join(sources)


class Stock(ContextCollection):
    """
    A collection of molecules that are in stock

    A molecule can be queried on the stock with:

    .. code-block::

        my_mol = Molecule(smiles="CCO")
        my_mol in stock

    One can obtain individual stocks with:

    .. code-block::

        sub_stock = stock["key"]

    One can obtain the number of molecules in the selected stock with:

    .. code-block::

        number_of_molecules = len(stock)

    """

    _collection_name = "stock"

    def __init__(self) -> None:
        super().__init__()
        self._exclude: Set[str] = set()
        self._stop_criteria: StrDict = {"amount": None, "price": None, "counts": {}}
        self._use_stop_criteria: bool = False

    def __contains__(self, mol: Molecule) -> bool:
        if not self.selection or mol.inchi_key in self._exclude:
            return False

        if self._use_stop_criteria:
            return self._apply_stop_criteria(mol)

        for key in self.selection:
            if mol in self[key]:
                return True
        return False

    def __len__(self) -> int:
        return sum(len(self[key]) for key in self.selection or [])

    @property
    def stop_criteria(self) -> dict:
        """Return a copy of the stop criteria used by the stock"""
        return copy.deepcopy(self._stop_criteria)

    def amount(self, mol: Molecule) -> float:
        """
        Calculate the maximum amount of a molecule in stock

        :param mol: the molecule to query
        :raises StockException: if the amount could not be computed
        :return: the maximum amount
        """
        amounts = self._mol_property(mol, "amount")
        if not amounts:
            raise StockException("Could not obtain amount of molecule")
        return max(amounts)

    def availability_string(self, mol: Molecule) -> str:
        """
        Return a string of what stocks a given mol is available

        If the molecule is not in stock it will return "Not in stock"

        :param mol: The molecule to query
        :returns: string with a list of stocks that mol was found in
        """
        availability = []
        for key in self.selection or []:
            if mol not in self[key]:
                continue
            try:
                availability.append(self[key].availability_string(mol))
            except (StockException, AttributeError):
                availability.append(key)
        if availability:
            return ",".join(availability)
        return "Not in stock"

    def exclude(self, mol: Molecule) -> None:
        """
        Exclude a molecule from the stock.
        When this molecule is queried it will return False,
        regardless if the molecule is in the stock.

        :param mol: the molecule to exclude
        """
        self._exclude.add(mol.inchi_key)

    def load(self, source: Union[str, Any], key: str) -> None:  # type: ignore
        """
        Load a stock.

        If `source` is a string, it is taken as a path to a filename and the
        stock is loaded as an `InMemoryInchiKeyQuery` object.

        If `source` is not a string, it is taken as a custom object that
        implements the `__contains__` and `__len__` methods for querying.

        :param source: the source of the sock
        :param key: The key that will be used to select the stock
        """
        src_str = str(source)
        if "object at 0x" in src_str:
            src_str = source.__class__.__name__
        self._logger.info(f"Loading stock from {src_str} to {key}")

        if isinstance(source, str):
            source = InMemoryInchiKeyQuery(source)
        self._items[key] = source

    def load_from_config(self, **config: Any) -> None:
        """
        Load stocks from a configuration

        The key can be "files" in case stocks are loaded from a file
        The key can be "mongodb" in case a ``MongoDbInchiKeyQuery`` object is instantiated
        The key can be "stop_criteria" in case the config is given to the `set_stop_criteria` method
        The key can point to a custom stock class, e.g. ``mypackage.mymodule.MyStock``
        in case this stock object is instantiated

        :param config: the configuration
        """
        known_keys = ["files", "mongodb", "stop_criteria"]
        if "stop_criteria" in config:
            self.set_stop_criteria(config["stop_criteria"])

        for key, stockfile in config.get("files", {}).items():
            self.load(stockfile, key)

        if "mongodb" in config:
            query_obj = MongoDbInchiKeyQuery(**(config["mongodb"] or {}))
            self.load(query_obj, "mongodb_stock")

        # Load stocks specifying a module and class, e.g. package.module.MyQueryClass
        for name, stock_config in config.items():
            if name in known_keys or name.find(".") == -1:
                continue

            module_name, class_name = name.rsplit(".", maxsplit=1)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                self._logger.warning(
                    f"Unable to load module '{module_name}' containing stock query classes"
                )
                continue

            if hasattr(module, class_name):
                query_obj = getattr(module, class_name)(**(stock_config or {}))
                self.load(query_obj, class_name)
            else:
                self._logger.warning(
                    f"Unable to find query class '{class_name}' in '{module_name}''"
                )

    def price(self, mol: Molecule) -> float:
        """
        Calculate the minimum price of a molecule in stock

        :param mol: the molecule to query
        :raises StockException: if the price could not be computed
        :return: the minimum price
        """
        prices = self._mol_property(mol, "price")
        if not prices:
            raise StockException("Could not obtain price of molecule")
        return min(prices)

    def reset_exclusion_list(self) -> None:
        """Remove all molecules in the exclusion list"""
        self._exclude = set()

    def select(self, value: Union[str, List[str]], append: bool = False) -> None:
        """
        Select one or more stock queries

        :param value: the key of the stocks to select
        :param append: if True and ``value`` is a single key append it to the current selection
        """
        super().select(value, append)
        try:
            self._logger.info(f"Compounds in stock: {len(self)}")
        except (TypeError, ValueError):  # In case len is not possible to compute
            pass

    def set_stop_criteria(self, criteria: dict = None) -> None:
        """
        Set criteria that stop the search

        The keys of the criteria can be "price" or "amount" which accepts numerical settings,
        or "counts" that should be dictionary of maximum allowed count for each atomic symbol.

        Example:

        .. code-block::

            criteria = {
                "price": 5,
                "amount": 100,
                "counts": {
                    "C": 6,
                    "O": 4
                }
            }

        :param criteria: the criteria settings
        """
        criteria = criteria or {}
        self._stop_criteria = {
            "price": criteria.get("price"),
            "amount": criteria.get("amount"),
            "counts": copy.deepcopy(criteria.get("size", criteria.get("counts"))),
        }
        self._use_stop_criteria = any(self._stop_criteria.values())
        reduced_criteria = {
            key: value for key, value in self._stop_criteria.items() if value
        }
        self._logger.info(f"Stop criteria for stock updated to: {reduced_criteria}")

    def smiles_in_stock(self, smiles: str) -> bool:
        """
        Check if the SMILES is in the currently selected stocks

        :param smiles: SMILES string (must be RDKit sanitizable)
        :returns: if the SMILES was on stock
        """
        return Molecule(smiles=smiles) in self

    def _apply_amount_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["amount"]:
            return True
        try:
            amount = self.amount(mol)
        except StockException:
            return True
        return amount >= self._stop_criteria.get("amount", amount)

    def _apply_counts_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["counts"]:
            return True
        atom_counts: dict = defaultdict(int)
        for atom in mol.rd_mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
        for symbol, threshold in self._stop_criteria["counts"].items():
            if atom_counts[symbol] > threshold:
                return False
        return True

    def _apply_price_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["price"]:
            return True
        try:
            price = self.price(mol)
        except StockException:
            return True
        return price <= self._stop_criteria.get("price", price)

    def _apply_stop_criteria(self, mol: Molecule) -> bool:
        if not self._apply_counts_criteria(mol):
            return False

        passes = False
        for key in self.selection or []:
            passes = passes or self[key].cached_search(mol)
        passes = passes and self._apply_amount_criteria(mol)
        passes = passes and self._apply_price_criteria(mol)

        for key in self.selection or []:
            self[key].clear_cache()
        return passes

    def _mol_property(self, mol, property_name):
        values = []
        for key in self.selection:
            try:
                func = getattr(self[key], property_name)
                values.append(func(mol))
            except StockException:
                pass
        return values
