.. meta::
  :description: rocThrust API data type support
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

Fundamental types
=================


 .. list-table:: Comparison of supported fundamental types of rocThrust and Thrust
    :header-rows: 1
    :name: supported-fundamental-types-rocthrust-vs-thrust

    *
      - Type
      - Size of type
      - rocThrust support
      - Thrust support
    *
      - :code:`unsigned char`
      - 8 bit
      - ✅
      - ✅
    *
      - :code:`unsigned short`
      - 16 bit
      - ✅
      - ✅
    *
      - :code:`unsigned int`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`unsigned long long`
      - 64 bit
      - ✅
      - ✅
    *
      - :code:`char`
      - 8 bit
      - ✅
      - ✅
    *
      - :code:`short`
      - 16 bit
      - ✅
      - ✅
    *
      - :code:`int`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`long long`
      - 64 bit
      - ✅
      - ✅
    *
      - :code:`half` [1]_
      - 16 bit
      - ⚠️
      - ⚠️
    *
      - :code:`float`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`double`
      - 64 bit
      - ✅
      - ✅

Custom types
============

rocThrust and Thrust support custom, user-defined types, if they provide the interface required by the used functions.

.. rubric:: Footnotes
.. [1] Limited support on the host side.
