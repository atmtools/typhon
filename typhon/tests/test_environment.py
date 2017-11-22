# -*- coding: utf-8 -*-
"""Testing the environment/configuration handler.
"""
import os
from copy import copy

import pytest

from typhon import environ
from typhon.config import conf


class TestEnvironment:
    """Testing the environment handler."""
    def setup_method(self):
        """Run all test methods with empty config and environment."""
        self.env = copy(os.environ)
        self.conf = copy(conf)

        os.environ = {}
        conf.clear()

    def teardown_method(self):
        """Restore config and environment."""
        os.environ = self.env
        conf = self.conf

    def test_get_environment_variables(self):
        """Test if environment variables are considered."""
        os.environ['TYPHON_ENV_TEST'] = 'TEST_VALUE'

        assert environ['TYPHON_ENV_TEST'] == 'TEST_VALUE'

    def test_set_environment_variables(self):
        """Test if environment variables are updated."""
        environ['TYPHON_ENV_TEST'] = 'TEST_VALUE'

        assert os.environ['TYPHON_ENV_TEST'] == 'TEST_VALUE'

    def test_undefined_variable(self):
        """Test behavior for undefined variables."""
        with pytest.raises(KeyError):
            dummy = environ['TYPHON_ENV_TEST']

    def test_membership(self):
        """Test the membership check."""
        os.environ['TYPHON_ENV_TEST'] = 'TEST_VALUE'

        assert 'TYPHON_ENV_TEST' in environ

    def test_membership_negative(self):
        """Test the membership check (negative)."""
        assert 'TYPHON_ENV_TEST' not in environ
