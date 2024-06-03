from typing import Any, List, Dict, Union, Collection

import pyarrow as pa
import pandas as pd

TableInput = Union[pd.DataFrame, Dict[str, List[Any]]]


class ArrowList:
    """
    A PyArrow dataset list, which supports faster appends than huggingface datasets, and can be used to get a
    Hugginface Dataset
    """
    def __init__(self, data: TableInput):
        self.table = pa.table(data)

    def append(self, new_data: TableInput):
        new_table = pa.table(new_data)
        self.table = pa.concat_tables([self.table, new_table], promote=True)

    def select(self, indices: Collection[int]):
        return [self[i] for i in indices]

    def __getitem__(self, index) -> Dict[str, Any]:
        if type(index) == slice:
            return self.table[index].to_pylist()
        else:
            return {col: self.table[col][index].as_py() for col in self.table.column_names}

    def __len__(self):
        return self.table.num_rows

    def __iter__(self):
        return (self[i] for i in range(len(self)))
