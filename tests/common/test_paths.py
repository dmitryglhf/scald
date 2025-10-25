from pathlib import Path

import pytest

from scald.common.paths import (
    ensure_output_dir,
    find_csv_in_project,
    get_project_root,
    resolve_csv_path,
    validate_file_access,
)


class TestResolveCsvPath:
    """Test resolve_csv_path function."""

    def test_absolute_path(self, tmp_path):
        """Should resolve absolute path correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2")

        result = resolve_csv_path(csv_file)
        assert result == csv_file.resolve()
        assert result.is_absolute()

    def test_string_path(self, tmp_path, monkeypatch):
        """Should accept and resolve string paths."""
        monkeypatch.chdir(tmp_path)
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        result = resolve_csv_path("data.csv")
        assert result == csv_file.resolve()
        assert isinstance(result, Path)

    def test_relative_path(self, tmp_path, monkeypatch):
        """Should resolve relative path from cwd."""
        monkeypatch.chdir(tmp_path)
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        result = resolve_csv_path("data.csv")
        assert result == csv_file.resolve()
        assert result.is_absolute()

    def test_relative_path_with_base_dir(self, tmp_path):
        """Should resolve relative path from specified base directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        csv_file = base_dir / "test.csv"
        csv_file.write_text("x,y\n1,2")

        result = resolve_csv_path("test.csv", base_dir=base_dir)
        assert result == csv_file.resolve()

    def test_expanduser(self, tmp_path, monkeypatch):
        """Should expand ~ to home directory."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        csv_file = home_dir / "data.csv"
        csv_file.write_text("a,b\n1,2")

        monkeypatch.setenv("HOME", str(home_dir))
        result = resolve_csv_path("~/data.csv")
        assert result == csv_file.resolve()

    def test_string_input(self, tmp_path):
        """Should accept string input."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col\n1")

        result = resolve_csv_path(str(csv_file))
        assert isinstance(result, Path)
        assert result.exists()

    def test_path_input(self, tmp_path):
        """Should accept Path input."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col\n1")

        result = resolve_csv_path(Path(csv_file))
        assert isinstance(result, Path)
        assert result.exists()

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        missing = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            resolve_csv_path(missing)

    def test_path_is_directory(self, tmp_path):
        """Should raise ValueError if path is a directory."""
        directory = tmp_path / "dir"
        directory.mkdir()

        with pytest.raises(ValueError, match="Path is not a file"):
            resolve_csv_path(directory)

    def test_non_csv_extension_warning(self, tmp_path):
        """Should still resolve non-CSV files (with warning)."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("data")

        result = resolve_csv_path(txt_file)
        assert result == txt_file.resolve()

    def test_unreadable_file(self, tmp_path):
        """Should raise PermissionError for unreadable file."""
        csv_file = tmp_path / "unreadable.csv"
        csv_file.write_text("data")
        csv_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError, match="File is not readable"):
                resolve_csv_path(csv_file)
        finally:
            csv_file.chmod(0o644)

    def test_symlink_resolution(self, tmp_path):
        """Should resolve symlinks to actual file."""
        actual_file = tmp_path / "actual.csv"
        actual_file.write_text("data")
        symlink = tmp_path / "link.csv"
        symlink.symlink_to(actual_file)

        result = resolve_csv_path(symlink)
        assert result == actual_file.resolve()


class TestFindCsvInProject:
    """Test find_csv_in_project function."""

    def test_find_in_data_dir(self, tmp_path, monkeypatch):
        """Should find CSV in data/ directory."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_file = data_dir / "test.csv"
        csv_file.write_text("data")

        result = find_csv_in_project("test.csv")
        assert result == csv_file.resolve()

    def test_find_in_examples_data(self, tmp_path, monkeypatch):
        """Should find CSV in examples/data/ directory."""
        monkeypatch.chdir(tmp_path)
        examples_data = tmp_path / "examples" / "data"
        examples_data.mkdir(parents=True)
        csv_file = examples_data / "iris.csv"
        csv_file.write_text("data")

        result = find_csv_in_project("iris.csv")
        assert result == csv_file.resolve()

    def test_find_in_current_dir(self, tmp_path, monkeypatch):
        """Should find CSV in current directory."""
        monkeypatch.chdir(tmp_path)
        csv_file = tmp_path / "local.csv"
        csv_file.write_text("data")

        result = find_csv_in_project("local.csv")
        assert result == csv_file.resolve()

    def test_priority_order(self, tmp_path, monkeypatch):
        """Should search in priority order: data/, examples/data/, etc."""
        monkeypatch.chdir(tmp_path)

        # Create file in multiple locations
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        data_csv = data_dir / "test.csv"
        data_csv.write_text("from data")

        current_csv = tmp_path / "test.csv"
        current_csv.write_text("from current")

        # Should find in data/ first
        result = find_csv_in_project("test.csv")
        assert result == data_csv.resolve()

    def test_custom_search_dirs(self, tmp_path, monkeypatch):
        """Should search in custom directories."""
        monkeypatch.chdir(tmp_path)
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        csv_file = custom_dir / "special.csv"
        csv_file.write_text("data")

        result = find_csv_in_project("special.csv", search_dirs=["custom"])
        assert result == csv_file.resolve()

    def test_not_found_returns_none(self, tmp_path, monkeypatch):
        """Should return None if file not found."""
        monkeypatch.chdir(tmp_path)

        result = find_csv_in_project("nonexistent.csv")
        assert result is None

    def test_find_file_not_directory(self, tmp_path, monkeypatch):
        """Should only match files, not directories."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_dir = data_dir / "test.csv"
        file_dir.mkdir()  # Create directory with .csv name

        result = find_csv_in_project("test.csv")
        assert result is None


class TestValidateFileAccess:
    """Test validate_file_access function."""

    def test_valid_file(self, tmp_path):
        """Should not raise for valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        validate_file_access(test_file)

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        missing = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_access(missing)

    def test_path_is_directory(self, tmp_path):
        """Should raise ValueError for directory."""
        directory = tmp_path / "dir"
        directory.mkdir()

        with pytest.raises(ValueError, match="Path is not a file"):
            validate_file_access(directory)

    def test_check_read_permission(self, tmp_path):
        """Should check read permission when requested."""
        test_file = tmp_path / "unreadable.txt"
        test_file.write_text("data")
        test_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError, match="File is not readable"):
                validate_file_access(test_file, check_read=True)
        finally:
            test_file.chmod(0o644)

    def test_check_write_permission(self, tmp_path):
        """Should check write permission when requested."""
        test_file = tmp_path / "readonly.txt"
        test_file.write_text("data")
        test_file.chmod(0o444)

        try:
            with pytest.raises(PermissionError, match="File is not writable"):
                validate_file_access(test_file, check_write=True)
        finally:
            test_file.chmod(0o644)

    def test_skip_read_check(self, tmp_path):
        """Should not check read permission if not requested."""
        test_file = tmp_path / "unreadable.txt"
        test_file.write_text("data")
        test_file.chmod(0o000)

        try:
            # Should not raise since check_read=False
            validate_file_access(test_file, check_read=False)
        finally:
            test_file.chmod(0o644)

    def test_both_permissions(self, tmp_path):
        """Should check both read and write permissions."""
        test_file = tmp_path / "rw.txt"
        test_file.write_text("data")
        test_file.chmod(0o600)

        # Should not raise for file with both permissions
        validate_file_access(test_file, check_read=True, check_write=True)


class TestEnsureOutputDir:
    """Test ensure_output_dir function."""

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directory if it doesn't exist."""
        output_file = tmp_path / "output" / "predictions" / "pred.csv"

        result = ensure_output_dir(output_file)

        assert output_file.parent.exists()
        assert output_file.parent.is_dir()
        assert result == output_file

    def test_existing_directory(self, tmp_path):
        """Should not fail if directory already exists."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        output_file = output_dir / "file.txt"

        result = ensure_output_dir(output_file)

        assert output_dir.exists()
        assert result == output_file

    def test_nested_directories(self, tmp_path):
        """Should create nested directory structure."""
        output_file = tmp_path / "a" / "b" / "c" / "d" / "file.csv"

        ensure_output_dir(output_file)

        assert output_file.parent.exists()
        assert (tmp_path / "a").exists()
        assert (tmp_path / "a" / "b").exists()
        assert (tmp_path / "a" / "b" / "c").exists()

    def test_returns_original_path(self, tmp_path):
        """Should return the original path for chaining."""
        output_file = tmp_path / "output" / "file.txt"

        result = ensure_output_dir(output_file)

        assert result == output_file
        assert isinstance(result, Path)


class TestGetProjectRoot:
    """Test get_project_root function."""

    def test_finds_git_marker(self, tmp_path, monkeypatch):
        """Should find project root with .git directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        git_dir = project_root / ".git"
        git_dir.mkdir()

        subdir = project_root / "src" / "module"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        result = get_project_root()
        assert result == project_root

    def test_finds_pyproject_marker(self, tmp_path, monkeypatch):
        """Should find project root with pyproject.toml."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("[tool.poetry]")

        subdir = project_root / "tests"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        result = get_project_root()
        assert result == project_root

    def test_finds_env_marker(self, tmp_path, monkeypatch):
        """Should find project root with .env file."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".env").write_text("KEY=value")

        subdir = project_root / "deep" / "nested" / "dir"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        result = get_project_root()
        assert result == project_root

    def test_multiple_markers(self, tmp_path, monkeypatch):
        """Should work with multiple markers present."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        (project_root / "pyproject.toml").write_text("")
        (project_root / "README.md").write_text("")

        monkeypatch.chdir(project_root)

        result = get_project_root()
        assert result == project_root

    def test_no_markers_returns_cwd(self, tmp_path, monkeypatch):
        """Should return cwd if no markers found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        result = get_project_root()
        assert result == empty_dir

    def test_stops_at_root(self, tmp_path, monkeypatch):
        """Should stop searching at filesystem root."""
        deep_dir = tmp_path / "very" / "deep" / "directory"
        deep_dir.mkdir(parents=True)
        monkeypatch.chdir(deep_dir)

        result = get_project_root()
        # Should return cwd since no markers found
        assert result == deep_dir
