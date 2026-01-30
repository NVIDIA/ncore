# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Package containing common and abstract functionality to implement NCore data-converters"""

from ncore.impl.data_converter.base import BaseDataConverter, BaseDataConverterConfig


__all__ = [
    "BaseDataConverter",
    "BaseDataConverterConfig",
]
