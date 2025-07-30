execute_process(COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/util/makeself/makeself.sh
                        --header ${CMAKE_CURRENT_LIST_DIR}/util/makeself/makeself-header.sh
                        --help-header ./help.info
                        --gzip --complevel 4 --nomd5 --sha256
                        ./ ${CPACK_PACKAGE_FILE_NAME} "version:1.0" ./install.sh
                WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
                RESULT_VARIABLE EXEC_RESULT
                ERROR_VARIABLE  EXEC_ERROR
)

if (NOT "${EXEC_RESULT}x" STREQUAL "0x")
  message(FATAL_ERROR "CPack Command error: ${EXEC_RESULT}\n${EXEC_ERROR}")
endif()

execute_process(COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES} ${CPACK_PACKAGE_DIRECTORY}/
        COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES} to ${CPACK_PACKAGE_DIRECTORY}/"
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )

if (NOT "${CPACK_PACKAGE_DIRECTORY}x" STREQUAL "${CPACK_INSTALL_PREFIX}x")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CPACK_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )

    execute_process(
        COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES} ${CPACK_INSTALL_PREFIX}/
        COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES} to ${CPACK_INSTALL_PREFIX}/"
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )
endif()


