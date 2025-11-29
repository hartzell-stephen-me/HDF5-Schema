# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of hdf5schema seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** disclose the vulnerability publicly until we have had a chance to address it

### Please Do

**Report security vulnerabilities via email:**

Send details to: **hartzell.stephen@gmail.com**

Include the following information:
- Type of vulnerability (e.g., code injection, denial of service, path traversal)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
2. **Investigation**: We will investigate and validate the reported vulnerability
3. **Communication**: We will keep you informed about our progress
4. **Resolution**: We will work on a fix and coordinate the release timeline with you
5. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Vulnerability Disclosure Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment sent
- **Day 3-7**: Vulnerability validated and severity assessed
- **Day 8-30**: Fix developed and tested
- **Day 31-90**: Fix released and public disclosure (coordinated with reporter)

We appreciate your help in keeping hdf5schema and its users safe!

## Security Considerations

### HDF5 File Processing

When using hdf5schema to validate HDF5 files:

1. **Untrusted Files**: Be cautious when validating HDF5 files from untrusted sources
   - HDF5 files can contain malicious data structures
   - Consider file size limits for untrusted inputs
   - Run validation in sandboxed environments for untrusted files

2. **Resource Limits**: Large or malformed HDF5 files can consume significant resources
   - Set appropriate timeouts for validation operations
   - Monitor memory usage when validating large files
   - Consider implementing file size limits

3. **Schema Injection**: Schemas are loaded from JSON files
   - Ensure schema files come from trusted sources
   - Validate schema format before use
   - Avoid loading schemas from user-controlled locations without validation

### Best Practices

1. **Input Validation**
   ```python
   # Always validate file paths
   import os
   if not os.path.exists(hdf5_file):
       raise ValueError("File does not exist")

   # Check file size before processing
   max_size = 100 * 1024 * 1024  # 100 MB
   if os.path.getsize(hdf5_file) > max_size:
       raise ValueError("File too large")
   ```

2. **Error Handling**
   ```python
   from hdf5schema import validate

   try:
       is_valid = validate(hdf5_file, schema)
   except Exception as e:
       # Log error details securely
       logger.error("Validation failed", exc_info=True)
       # Don't expose internal details to users
       raise ValueError("Validation failed")
   ```

3. **Schema Security**
   ```python
   import json
   from pathlib import Path

   # Load schemas from trusted locations only
   schema_dir = Path("/trusted/schemas/")
   schema_file = schema_dir / "my_schema.json"

   if not schema_file.is_relative_to(schema_dir):
       raise ValueError("Schema path traversal detected")

   with open(schema_file) as f:
       schema = json.load(f)
   ```

## Known Security Limitations

### 1. HDF5 Library Dependencies

hdf5schema depends on the `h5py` library, which in turn depends on the HDF5 C library:
- Security vulnerabilities in h5py or HDF5 C library may affect hdf5schema
- Keep dependencies updated: `pip install --upgrade h5py`
- Monitor security advisories for h5py and HDF5

### 2. Schema Complexity

Complex schemas with deep nesting or recursive references:
- May consume significant memory during validation
- Could potentially cause stack overflow in extreme cases
- Consider limiting schema complexity for untrusted inputs

### 3. Regular Expression Patterns

Schemas can include regex patterns for matching:
- Poorly written regex can cause ReDoS (Regular Expression Denial of Service)
- We use Python's `re` module which has some protections
- Still recommended to validate regex patterns in schemas from untrusted sources

## Security Updates

Security updates will be released as patch versions (e.g., 1.0.1, 1.0.2) and will be clearly marked in:
- GitHub Releases with "Security" tag
- [CHANGELOG.md](CHANGELOG.md) with "Security" prefix
- Security advisories on GitHub

## Dependencies

hdf5schema has the following dependencies:
- h5py (HDF5 file I/O)
- numpy (array operations)
- jsonschema (JSON schema validation)
- typer (CLI interface)

We monitor security advisories for all dependencies and update as needed. You can check for outdated dependencies:

```bash
pip list --outdated
```

Update dependencies:
```bash
pip install --upgrade hdf5schema
```

## Additional Resources

- [HDF5 Security](https://www.hdfgroup.org/HDF5/)
- [h5py Documentation](https://docs.h5py.org/)
- [OWASP Input Validation](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)

## Contact

For security-related questions or concerns:
- Email: hartzell.stephen@gmail.com
- GitHub Issues (for non-security bugs): https://github.com/hartzell-stephen-me/hdf5schema/issues

Thank you for helping keep hdf5schema secure!
