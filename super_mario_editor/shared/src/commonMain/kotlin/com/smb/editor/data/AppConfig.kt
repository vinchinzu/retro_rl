package com.smb.editor.data

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class AppConfig(
    val windowX: Int = 100,
    val windowY: Int = 100,
    val windowWidth: Int = 1280,
    val windowHeight: Int = 800,
    val lastRomPath: String? = null,
    val lastLevel: String? = null
) {
    companion object {
        private val json = Json { prettyPrint = true; ignoreUnknownKeys = true }

        fun fromJson(text: String): AppConfig = json.decodeFromString(text)
    }

    fun toJson(): String = Companion.json.encodeToString(serializer(), this)
}
