"""
core/ingestion/data_loader.py

Responsible for loading, validating, and schema-profiling raw CSV data
into pandas DataFrames. Strictly core-layer: no UI, no I/O side effects,
no print statements.
"""

from __future__ import annotations

import io
from typing import Union

import pandas as pd


class DataLoader:
    """
    Ingests raw CSV data into validated pandas DataFrames and exposes
    lightweight schema detection for downstream pipeline stages.

    All methods are stateless — each call is self-contained and
    produces no side effects beyond its return value.
    """

    # Encodings attempted in priority order before raising.
    _ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_csv(self, file: Union[str, bytes, io.IOBase]) -> pd.DataFrame:
        """
        Load a CSV file into a validated DataFrame.

        Attempts multiple encodings automatically to handle files that are
        not UTF-8 encoded. Raises descriptive errors rather than propagating
        low-level pandas/IO exceptions to callers.

        Parameters
        ----------
        file:
            Accepted forms:
            - A filesystem path (``str``).
            - Raw bytes (``bytes``) — e.g. from an uploaded file buffer.
            - Any file-like object that supports ``.read()`` (``io.IOBase``
              and its subclasses, including ``io.BytesIO`` / ``io.StringIO``).

        Returns
        -------
        pd.DataFrame
            A non-empty, validated DataFrame parsed from the CSV source.

        Raises
        ------
        ValueError
            If the input type is unsupported, the file cannot be decoded
            with any known encoding, or the resulting DataFrame is empty /
            structurally invalid.
        """
        raw: Union[str, bytes] = self._read_raw(file)
        df: pd.DataFrame = self._parse_with_encoding_fallback(raw)
        self.validate_dataframe(df)
        return df

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Assert that a DataFrame meets the minimum structural requirements
        for downstream processing.

        Parameters
        ----------
        df:
            DataFrame to validate.

        Raises
        ------
        ValueError
            If ``df`` is not a ``pd.DataFrame``, has no columns, or has
            no rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Expected a pd.DataFrame, received {type(df).__name__!r}."
            )
        if df.shape[1] == 0:
            raise ValueError("Dataset is invalid: contains zero columns.")
        if df.shape[0] == 0:
            raise ValueError("Dataset is empty: contains zero rows.")

    def detect_schema(self, df: pd.DataFrame) -> dict:
        """
        Derive a lightweight structural schema from a validated DataFrame.

        Uses vectorised pandas operations exclusively — no Python-level
        loops over columns or rows.

        Parameters
        ----------
        df:
            A non-empty DataFrame (pre-validated via :meth:`validate_dataframe`).

        Returns
        -------
        dict
            Schema mapping with the following keys:

            ``rows`` : int
                Total number of rows.
            ``columns`` : int
                Total number of columns.
            ``numeric_columns`` : list[str]
                Columns whose dtype is numeric (int or float variants).
            ``categorical_columns`` : list[str]
                Columns whose dtype is ``object`` or ``category``.
            ``datetime_columns`` : list[str]
                Columns whose dtype is a datetime variant.
            ``memory_usage_mb`` : float
                Deep memory footprint of the DataFrame in megabytes,
                rounded to four decimal places.
        """
        self.validate_dataframe(df)

        dtypes = df.dtypes

        numeric_mask     = dtypes.map(pd.api.types.is_numeric_dtype)
        datetime_mask    = dtypes.map(pd.api.types.is_datetime64_any_dtype)
        categorical_mask = dtypes.map(
            lambda dt: (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                # pd.StringDtype (pandas >= 1.0) is string but not object or numeric
                or (pd.api.types.is_string_dtype(dt) and not pd.api.types.is_numeric_dtype(dt))
            )
        )

        memory_bytes: int = df.memory_usage(deep=True).sum()

        return {
            "rows":                int(df.shape[0]),
            "columns":             int(df.shape[1]),
            "numeric_columns":     dtypes.index[numeric_mask].tolist(),
            "categorical_columns": dtypes.index[categorical_mask].tolist(),
            "datetime_columns":    dtypes.index[datetime_mask].tolist(),
            "memory_usage_mb":     round(memory_bytes / (1024 ** 2), 4),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_raw(file: Union[str, bytes, io.IOBase]) -> Union[str, bytes]:
        """
        Normalise the *file* argument to either a path string or raw bytes.

        Parameters
        ----------
        file:
            Path string, bytes, or file-like object.

        Returns
        -------
        str or bytes
            A filesystem path (str) or raw byte content (bytes).

        Raises
        ------
        ValueError
            If the type of *file* is not supported or reading fails.
        """
        if isinstance(file, str):
            return file  # filesystem path — pass directly to pandas

        if isinstance(file, bytes):
            return file

        if isinstance(file, io.IOBase) or hasattr(file, "read"):
            try:
                content = file.read()
            except Exception as exc:
                raise ValueError(
                    f"Failed to read from file-like object: {exc}"
                ) from exc
            if isinstance(content, str):
                # StringIO — encode to bytes so the fallback loop is uniform
                return content.encode("utf-8")
            if isinstance(content, bytes):
                return content
            raise ValueError(
                f"file.read() returned unexpected type {type(content).__name__!r}."
            )

        raise ValueError(
            f"Unsupported file type {type(file).__name__!r}. "
            "Provide a path string, bytes, or a file-like object."
        )

    def _parse_with_encoding_fallback(
        self, raw: Union[str, bytes]
    ) -> pd.DataFrame:
        """
        Attempt to parse *raw* as CSV, cycling through :attr:`_ENCODINGS`.

        Parameters
        ----------
        raw:
            Filesystem path (str) or raw byte content (bytes).

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If all encoding attempts fail or the parsed result is not a
            DataFrame (e.g. pandas returns a ``TextFileReader``).
        """
        last_error: Exception | None = None

        if isinstance(raw, str):
            # Filesystem path — try each encoding in sequence.
            for encoding in self._ENCODINGS:
                try:
                    df = pd.read_csv(raw, encoding=encoding)
                    return self._coerce_dataframe(df)
                except (UnicodeDecodeError, LookupError):
                    continue
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse CSV file at path {raw!r}: {exc}"
                    ) from exc
        else:
            # Byte content — wrap in a fresh BytesIO for each attempt.
            for encoding in self._ENCODINGS:
                try:
                    buf = io.BytesIO(raw)
                    df = pd.read_csv(buf, encoding=encoding)
                    return self._coerce_dataframe(df)
                except (UnicodeDecodeError, LookupError) as exc:
                    last_error = exc
                    continue
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse CSV content: {exc}"
                    ) from exc

        raise ValueError(
            f"Unable to decode CSV with any of the attempted encodings "
            f"({', '.join(self._ENCODINGS)}). Last error: {last_error}"
        )

    @staticmethod
    def _coerce_dataframe(obj: object) -> pd.DataFrame:
        """
        Ensure *obj* is a concrete ``pd.DataFrame``.

        ``pd.read_csv`` can return a ``TextFileReader`` when called with
        ``chunksize``; this guard prevents that from silently propagating.

        Parameters
        ----------
        obj:
            Object returned by ``pd.read_csv``.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If *obj* is not a ``pd.DataFrame``.
        """
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(
                f"CSV parsing produced {type(obj).__name__!r} instead of a "
                "DataFrame. Ensure the file is a plain, non-chunked CSV."
            )
        return obj