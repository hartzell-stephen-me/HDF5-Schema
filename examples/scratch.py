from jsonschema import validate


meta_schema = {
    "type": "object",
    "properties": {
        "type": {
            "pattern": "dataset"
        },
        "description" : {
            "type": "string"
        },
        "dtype": {
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "dtype": {
                            "type": "string"
                        },
                        "offset": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "name",
                        "dtype"
                    ]
                },
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "dtype": {
                                "type": "string"
                            },
                            "offset": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "name",
                            "dtype"
                        ]
                    }
                }
            ]
        },
        "attrs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                                "type": "string"
                    },
                    "dtype": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string"
                                    },
                                    "dtype": {
                                        "type": "string"
                                    }
                                }
                            },
                            {
                                "type": "array",
                                "items": {
                                    "name": {
                                        "type": "string"
                                    },
                                    "dtype": {
                                        "type": "string"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        },
        "shape": {
            "type": "array"
        }
    },
    "required": [
        "type"
    ]
}

schema = {
    "type": "dataset",
    "description": "Attachments",
}

validate(instance=schema, schema=meta_schema)
