# -*- coding: utf-8 -*-
"""Testing the environment/configuration handler.
"""
import os
from copy import copy

import pytest

from typhon import environment


class TestEnvironment:
    """Testing the environment handler."""
    def setup_method(self):
        """Run all test methods with an empty environment."""
        self.env = copy(os.environ)
        os.environ = {}

    def teardown_method(self):
        """Restore old environment."""
        os.environ = self.env

    def test_get_environment_variables(self):
        """Test if environment variables are considered."""
        os.environ['TYPHON_ENV_TEST'] = 'TEST_VALUE'

        assert environment.environ['TYPHON_ENV_TEST'] == 'TEST_VALUE'

    def test_set_environment_variables(self):
        """Test if environment variables are updated."""
        environment.environ['TYPHON_ENV_TEST'] = 'TEST_VALUE'

        assert os.environ['TYPHON_ENV_TEST'] == 'TEST_VALUE'

    def test_undefined_variable(self):
        """Test behavior for undefined variables."""
        with pytest.raises(KeyError):
            environment.environ['TYPHON_ENV_TEST']
